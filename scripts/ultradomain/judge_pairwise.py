from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any, Dict, List, Tuple

from relrag.utils.openai_client import chat_completion
from scripts.ultradomain.common import (
    ANSWERS_DIR,
    DOMAIN_LABELS,
    JUDGE_DIR,
    QUESTIONS_DIR,
    RUN_META_DIR,
    SUMMARY_DIR,
    ensure_dirs,
    extract_json_object,
    get_api_key,
    now_iso,
    read_json,
    read_jsonl,
    sha256_text,
    ultradomain_get,
    write_json,
)

PAIRS = [
    ("RelRAG-full", "BM25-only"),
    ("RelRAG-full", "Dense-only"),
    ("RelRAG-full", "Hybrid-only"),
]

DIMENSIONS = ["comprehensiveness", "diversity", "empowerment", "overall"]

SYSTEM_PROMPT = "你是一个严格的RAG回答评估员。"

USER_PROMPT_TEMPLATE = """请比较以下两份答案，分别在四个维度上判断更好者：
- Comprehensiveness: 覆盖面
- Diversity: 信息多样性/角度
- Empowerment: 是否有助于用户理解和后续行动
- Overall: 综合

如果两者表现相当，请判为 tie。

问题：
{question}

答案A：
{answer_a}

答案B：
{answer_b}

输出严格 JSON：{{
  "winner": {{
    "comprehensiveness": "A"|"B"|"tie",
    "diversity": "A"|"B"|"tie",
    "empowerment": "A"|"B"|"tie",
    "overall": "A"|"B"|"tie"
  }},
  "reason": "简短理由"
}}"""


def _json_hash(payload: Any) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _read_optional_json(path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_protocol_version() -> str | None:
    path = RUN_META_DIR / "protocol_version"
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _load_answers(domain: str, system: str) -> Dict[str, Dict[str, Any]]:
    path = ANSWERS_DIR / domain / f"{system}.jsonl"
    answers: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = row.get("question_id")
        if qid:
            answers[qid] = row
    return answers


def _question_parity(qid: str) -> int:
    nums = re.findall(r"\d+", str(qid))
    if not nums:
        return 0
    return int(nums[-1]) % 2


def _call_judge(
    messages: List[Dict[str, str]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    return chat_completion(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"top_p": top_p},
    )


def _call_json_object(
    messages: List[Dict[str, str]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    retries: int = 2,
    backoff_sec: float = 1.0,
) -> Tuple[Dict[str, Any] | None, str | None, str | None]:
    last_err: str | None = None
    last_content: str | None = None
    for attempt in range(retries + 1):
        try:
            content = _call_judge(
                messages,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            last_content = content
            obj = extract_json_object(content)
            return obj, content, None
        except Exception as exc:  # noqa: PERF203
            last_err = str(exc)
            if attempt < retries:
                time.sleep(backoff_sec * (2 ** attempt))
                continue
            break
    return None, last_content, last_err


def _compute_win_rates(rows: List[Dict[str, Any]], system_left: str) -> Dict[str, Any]:
    totals = {dim: {"wins": 0, "ties": 0, "total": 0} for dim in DIMENSIONS}
    for row in rows:
        winner = row.get("winner") or {}
        order = row.get("order") or {}
        a_sys = order.get("A")
        b_sys = order.get("B")
        for dim in DIMENSIONS:
            decision = (winner.get(dim) or "").strip()
            if decision not in {"A", "B", "tie"}:
                continue
            totals[dim]["total"] += 1
            if decision == "tie":
                totals[dim]["ties"] += 1
            elif decision == "A" and a_sys == system_left:
                totals[dim]["wins"] += 1
            elif decision == "B" and b_sys == system_left:
                totals[dim]["wins"] += 1

    summary = {}
    for dim, values in totals.items():
        total = values["total"]
        wins = values["wins"]
        ties = values["ties"]
        win_rate = (wins + 0.5 * ties) / total if total else 0.0
        summary[dim] = {
            "wins": wins,
            "ties": ties,
            "total": total,
            "win_rate": round(win_rate, 4),
        }
    return summary


def main() -> None:
    domain_default = ultradomain_get("dataset.domain", "all")
    base_url_default = ultradomain_get("llm.base_url", "https://api.deepseek.com/v1")
    model_default = ultradomain_get("llm.model", "deepseek-chat")
    api_key_env_default = ultradomain_get("llm.api_key_env", "DEEPSEEK_API_KEY")
    judge_temp_default = float(ultradomain_get("judge.temperature", 0.0) or 0.0)
    judge_top_p_default = float(ultradomain_get("judge.top_p", 1.0) or 1.0)
    judge_max_tokens_default = int(ultradomain_get("judge.max_output_tokens", 1024) or 1024)
    resume_default = bool(ultradomain_get("pipeline.resume", False))

    parser = argparse.ArgumentParser(description="Judge UltraDomain pairwise comparisons.")
    parser.add_argument("--domain", default=domain_default)
    parser.add_argument("--base_url", default=base_url_default)
    parser.add_argument("--model", default=model_default)
    parser.add_argument("--api_key_env", default=api_key_env_default)
    parser.add_argument("--temperature", type=float, default=judge_temp_default)
    parser.add_argument("--top_p", type=float, default=judge_top_p_default)
    parser.add_argument("--max_output_tokens", type=int, default=judge_max_tokens_default)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=resume_default)
    args = parser.parse_args()

    ensure_dirs()
    api_key = get_api_key(args.api_key_env)
    protocol_version = _read_protocol_version()
    run_cfg = _read_optional_json(RUN_META_DIR / "run_config.json")
    system_cfg = _read_optional_json(RUN_META_DIR / "system_configs.json")
    prompt_template_hash = sha256_text(SYSTEM_PROMPT + "\n" + USER_PROMPT_TEMPLATE)
    domains = list(DOMAIN_LABELS.keys()) if args.domain == "all" else [args.domain]

    summary_lines: List[str] = ["# Win-Rate Summary", ""]

    for domain in domains:
        if domain not in DOMAIN_LABELS:
            continue
        questions = read_json(QUESTIONS_DIR / f"questions_{domain}.json").get("questions") or []
        for left, right in PAIRS:
            out_dir = JUDGE_DIR / domain
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{left}_vs_{right}.jsonl"
            meta_path = out_dir / f"{left}_vs_{right}.meta.json"
            existing = set()
            if args.resume and out_path.exists():
                for row in read_jsonl(out_path):
                    key = row.get("question_id")
                    if key:
                        existing.add(key)

            answers_left = _load_answers(domain, left)
            answers_right = _load_answers(domain, right)

            rows: List[Dict[str, Any]] = []
            for q in questions:
                qid = q.get("question_id")
                question = q.get("question")
                if not qid or not question:
                    continue
                if qid in existing:
                    continue
                a = answers_left.get(qid)
                b = answers_right.get(qid)
                if not a or not b:
                    continue
                flip = _question_parity(qid) == 0
                if flip:
                    order = {"A": right, "B": left}
                    answer_a = b.get("answer_final") or b.get("answer_raw")
                    answer_b = a.get("answer_final") or a.get("answer_raw")
                else:
                    order = {"A": left, "B": right}
                    answer_a = a.get("answer_final") or a.get("answer_raw")
                    answer_b = b.get("answer_final") or b.get("answer_raw")

                prompt = USER_PROMPT_TEMPLATE.format(question=question, answer_a=answer_a, answer_b=answer_b)
                prompt_instance_hash = _json_hash(
                    {
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": prompt,
                        "model": args.model,
                        "base_url": args.base_url,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_output_tokens": args.max_output_tokens,
                    }
                )
                obj, content, err = _call_json_object(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=args.model,
                    base_url=args.base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_output_tokens,
                )
                rows.append(
                    {
                        "question_id": qid,
                        "domain": domain,
                        "order": order,
                        "winner": obj.get("winner") if obj else None,
                        "reason": obj.get("reason") if obj else None,
                        "order_flag": "swap" if flip else "normal",
                        "prompt_template_hash": prompt_template_hash,
                        "prompt_instance_hash": prompt_instance_hash,
                        "run_meta": {
                            "domain": domain,
                            "left_system": left,
                            "right_system": right,
                            "protocol_version": protocol_version,
                            "model": args.model,
                            "base_url": args.base_url,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_output_tokens": args.max_output_tokens,
                            "dimensions": DIMENSIONS,
                            "tie_policy": "0.5_each_side",
                            "ab_bias_control": "qid_parity_swap",
                            "config_snapshot": {
                                "judge_llm": {
                                    "model": args.model,
                                    "base_url": args.base_url,
                                    "temperature": args.temperature,
                                    "top_p": args.top_p,
                                    "max_output_tokens": args.max_output_tokens,
                                },
                                "protocol_run_config": run_cfg,
                                "protocol_system_config": system_cfg,
                            },
                            "generated_at": now_iso(),
                        },
                        "raw_output": content,
                        "error": err,
                        "generated_at": now_iso(),
                    }
                )

            if rows:
                mode = "a" if args.resume and out_path.exists() else "w"
                with out_path.open(mode, encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False))
                        handle.write("\n")

            all_rows = list(read_jsonl(out_path)) if out_path.exists() else []
            write_json(
                meta_path,
                {
                    "domain": domain,
                    "left_system": left,
                    "right_system": right,
                    "protocol_version": protocol_version,
                    "prompt_template_hash": prompt_template_hash,
                    "judge_llm": {
                        "model": args.model,
                        "base_url": args.base_url,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_output_tokens": args.max_output_tokens,
                    },
                    "config_snapshot": {
                        "protocol_run_config": run_cfg,
                        "protocol_system_config": system_cfg,
                    },
                    "row_count": len(all_rows),
                    "updated_at": now_iso(),
                },
            )
            stats = _compute_win_rates(all_rows, left)
            summary_lines.append(f"## {domain} - {left} vs {right}")
            summary_lines.append("")
            summary_lines.append("| Dimension | Wins | Ties | Total | Win Rate |")
            summary_lines.append("| --- | --- | --- | --- | --- |")
            for dim in DIMENSIONS:
                entry = stats.get(dim) or {}
                summary_lines.append(
                    f"| {dim} | {entry.get('wins', 0)} | {entry.get('ties', 0)} | {entry.get('total', 0)} | {entry.get('win_rate', 0.0)} |"
                )
            summary_lines.append("")

    summary_path = SUMMARY_DIR / "summary_winrate.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
