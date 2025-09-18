"""
DBTES: Double-round Battle Tournament with Early Stopping
对战式双轮锦标赛边选择算法（向后兼容：支持仅 {id,score} 的候选）
"""
from typing import List, Dict, Any, Callable
import re
from .anchors import extract_anchors

# ---------- 小工具 ----------
def _snip(txt: str, L: int = 80) -> str:
    return (txt or "")[:L]

def _get_note(nid: int,
              notes_dict: Dict[int, Dict[str, Any]] | None,
              note_lookup: Callable[[int], Dict[str, Any]] | None) -> Dict[str, Any]:
    """优先从 notes_dict 取；否则用 note_lookup；最后兜底占位"""
    n = None
    if notes_dict:
        n = notes_dict.get(nid)
    if (not n) and note_lookup:
        try:
            n = note_lookup(nid)
        except Exception:
            n = None
    if not n:
        n = {"id": nid, "title": f"Note {nid}", "text": ""}
    n.setdefault("title", f"Note {nid}")
    n.setdefault("text", "")
    return n

# ---------- Prompt 生成 ----------
def _create_prompt(center_note: Dict[str, Any],
                   candidates: List[Dict[str, Any]],
                   config: Dict[str, Any],
                   notes_dict: Dict[int, Dict[str, Any]] | None = None,
                   note_lookup: Callable[[int], Dict[str, Any]] | None = None,
                   is_layer2: bool = False) -> str:
    """轻量 Prompt；Layer-2 增加片段长度"""
    snip_len = 130 if is_layer2 else 80
    c_title = center_note.get("title", f"Note {center_note.get('id')}")
    c_text  = center_note.get("text", "")
    c_snip  = _snip(c_text, snip_len)
    c_anchors = extract_anchors(c_title + " " + c_text)[:2]
    anchors_str = ", ".join(c_anchors) if c_anchors else "None"

    parts = [
        "SYSTEM: You compare candidates and pick the single best bridge.",
        f"CENTER: {c_title} — {c_snip}",
        f"ANCHORS: {anchors_str}",
        ""
    ]
    for i, cand in enumerate(candidates, 1):
        nid = cand["id"]
        n = _get_note(nid, notes_dict, note_lookup)
        parts.append(f"CANDIDATE {i}: {n.get('title','')} — {_snip(n.get('text',''), snip_len)}")
    parts += [
        "",
        "RULES: Prefer candidates sharing the same person/place/time as CENTER, "
        "or obviously complementary relation (e.g., spouse/partner of the performer).",
        "OUTPUT: WIN: CANDIDATE X (because ...)"
    ]
    return "\n".join(parts)

# ---------- 解析 LLM 输出 ----------
def _parse_llm_response(response: str,
                        group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """解析 WIN 行；找不到则返回空，由上层回退"""
    winners = []
    if not response:
        return winners
    for line in response.splitlines():
        line = line.strip()
        if not line.lower().startswith("win:"):
            continue
        m = re.search(r"candidate\s+(\d+)", line, re.IGNORECASE)
        if not m:
            continue
        k = int(m.group(1)) - 1
        if 0 <= k < len(group):
            reason_m = re.search(r"\(([^)]+)\)", line)
            reason = (reason_m.group(1) if reason_m else "Selected by LLM")[:100]
            cand = group[k]
            winners.append({
                "id": cand["id"],
                "score": cand.get("score", 0.0),
                "reason": reason
            })
    return winners

# ---------- 单轮锦标赛 ----------
def _tournament_round(center_note: Dict[str, Any],
                      candidates: List[Dict[str, Any]],
                      llm_call: Callable[[str], str],
                      config: Dict[str, Any],
                      notes_dict: Dict[int, Dict[str, Any]] | None = None,
                      note_lookup: Callable[[int], Dict[str, Any]] | None = None,
                      is_layer2: bool = False) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    m = int(config.get("m", 6))
    winners: List[Dict[str, Any]] = []

    for i in range(0, len(candidates), m):
        group = candidates[i:i+m]
        if len(group) == 1:
            winners.append({
                "id": group[0]["id"],
                "score": group[0].get("score", 0.0),
                "reason": "Auto-advance (single candidate)"
            })
            continue

        try:
            prompt = _create_prompt(center_note, group, config,
                                    notes_dict=notes_dict, note_lookup=note_lookup,
                                    is_layer2=is_layer2)
            resp = llm_call(prompt)
            sel = _parse_llm_response(resp, group)
            if sel:
                # 每组最多录 1~2 个胜者
                winners.extend(sel[:min(2, len(group))])
            else:
                # 回退：按分数取 1 个
                g = sorted(group, key=lambda x: x.get("score", 0.0), reverse=True)
                winners.append({"id": g[0]["id"], "score": g[0].get("score", 0.0),
                                "reason": "LLM parsing failed, score fallback"})
        except Exception as e:
            g = sorted(group, key=lambda x: x.get("score", 0.0), reverse=True)
            winners.append({"id": g[0]["id"], "score": g[0].get("score", 0.0),
                            "reason": f"LLM call failed: {str(e)[:50]}"})
    return winners

# ---------- 公开入口 ----------
def dbtes_select_edges(center_note: Dict[str, Any],
                       candidates: List[Dict[str, Any]],
                       llm_call: Callable[[str], str],
                       config: Dict[str, Any],
                       *,
                       notes: List[Dict[str, Any]] | None = None,
                       note_lookup: Callable[[int], Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    """
    输入:
      - center_note: {id,title,text}
      - candidates: [{id,score}]（可不含 title/text）
      - llm_call: callable(prompt)->str
      - config: {m,rounds,keep_k,max_tokens_per_call}
      - notes: 可选 [{id,title,text,...}]
      - note_lookup: 可选 id->note 的回调（当 notes 缺失时兜底）
    输出:
      - [{id, reason}]，长度 <= keep_k
    """
    if not candidates:
        return []

    keep_k = int(config.get("keep_k", 3))
    rounds = int(config.get("rounds", 2))

    # 限定候选上限以控成本
    if len(candidates) > 20:
        candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:20]

    # 构建 notes_dict（优先用 notes；否则从 candidates 自带字段；再否则靠 note_lookup）
    notes_dict: Dict[int, Dict[str, Any]] = {}
    if notes:
        notes_dict.update({n["id"]: n for n in notes})
    else:
        for c in candidates:
            if ("title" in c) or ("text" in c):
                notes_dict[c["id"]] = {"id": c["id"],
                                       "title": c.get("title", f"Note {c['id']}"),
                                       "text": c.get("text", "")}

    curr = candidates[:]
    for r in range(rounds):
        if len(curr) <= keep_k:
            break
        curr = _tournament_round(center_note, curr, llm_call, config,
                                 notes_dict=notes_dict, note_lookup=note_lookup,
                                 is_layer2=(r > 0))
        if not curr:
            # 兜底：按分数直接取 keep_k
            tmp = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:keep_k]
            return [{"id": c["id"], "reason": "Tournament failed, score fallback"} for c in tmp]

    # 截断并保证 reason 存在
    out = curr[:keep_k]
    for e in out:
        e.setdefault("reason", "Tournament winner")
    return out
