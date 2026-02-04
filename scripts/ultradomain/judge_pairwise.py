from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List

from relrag.utils.openai_client import chat_completion
from scripts.ultradomain.common import (
    ANSWERS_DIR,
    DOMAIN_LABELS,
    JUDGE_DIR,
    QUESTIONS_DIR,
    SUMMARY_DIR,
    ensure_dirs,
    extract_json_object,
    get_api_key,
    now_iso,
    read_json,
    read_jsonl,
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


def _call_judge(messages: List[Dict[str, str]], *, model: str, base_url: str, api_key: str) -> str:
    return chat_completion(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_tokens=1024,
        extra_body={"top_p": 1.0},
    )


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
    parser = argparse.ArgumentParser(description="Judge UltraDomain pairwise comparisons.")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--base_url", default="https://api.deepseek.com/v1")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    api_key = get_api_key(args.api_key_env)
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
                content = _call_judge(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=args.model,
                    base_url=args.base_url,
                    api_key=api_key,
                )
                obj = extract_json_object(content)
                rows.append(
                    {
                        "question_id": qid,
                        "domain": domain,
                        "order": order,
                        "winner": obj.get("winner"),
                        "reason": obj.get("reason"),
                        "order_flag": "swap" if flip else "normal",
                        "raw_output": content,
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
