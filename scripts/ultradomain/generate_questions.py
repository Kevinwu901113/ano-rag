from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

from relrag.utils.openai_client import chat_completion
from scripts.ultradomain.common import (
    DOMAIN_LABELS,
    QUESTIONS_DIR,
    RUN_META_DIR,
    TOKENIZER_ID,
    ensure_dirs,
    extract_json_list,
    extract_json_object,
    get_api_key,
    now_iso,
    read_json,
    sha256_text,
    write_json,
)

SYSTEM_PROMPT = "你是一个评测集生成器。你的任务是为给定语料库生成高层次\u201csensemaking\u201d问题集合。"

USER_PROMPT_TEMPLATE = """给你一段“数据集描述 + 语料库摘要（可选）”。请生成：

五个 RAG 用户（每个含：name/背景/专长/动机/提问风格）

每个用户五个任务（每个任务含：目标/约束/关心点）

每个（用户，任务）生成五个问题：必须需要跨文档/跨段落综合，避免单段 factoid；问题应可用语料回答但需要全局理解。

输出严格 JSON：users[5]{profile,tasks[5]{task_desc,questions[5]}}。

数据集描述：
{dataset_desc}

语料库摘要（可选）：
{summary}
"""

VALIDATOR_SYSTEM = "你是一个严格的评测集质量检查器。"
VALIDATOR_USER_TEMPLATE = """判断每个问题是否需要跨文档/跨段落综合（multi-doc），还是单段事实问题（single-factoid）。

数据集描述：
{dataset_desc}

问题列表：
{questions_block}

输出严格 JSON：{{\"labels\": [\"multi-doc\"|\"single-factoid\", ...]}}，长度必须与问题数量一致。"""

REWRITE_SYSTEM = "你是一个评测集生成器，负责把问题改写为需要跨文档综合的高层次问题。"
REWRITE_USER_TEMPLATE = """请将以下问题改写为 multi-doc 问题，仍可由该语料库回答，但需要跨文档/跨段落综合。保持原意方向，避免变成单句事实。

数据集描述：
{dataset_desc}

问题列表：
{questions_block}

输出严格 JSON：{{\"questions\": [\"...\", \"...\"]}}，长度必须与问题数量一致。"""


def _build_dataset_desc(domain: str) -> str:
    doc_stats_path = RUN_META_DIR / "doc_stats.json"
    chunk_stats_path = QUESTIONS_DIR.parent / "chunks" / "chunk_stats.json"
    desc_lines: List[str] = [f"Domain: {DOMAIN_LABELS.get(domain, domain)} ({domain})"]
    if doc_stats_path.exists():
        stats = read_json(doc_stats_path)
        dom = (stats.get("domains") or {}).get(domain) or {}
        if dom:
            desc_lines.append(f"Documents: {dom.get('doc_count', 0)}")
            desc_lines.append(f"Avg doc chars: {dom.get('avg_chars', 0)}")
    if chunk_stats_path.exists():
        stats = read_json(chunk_stats_path)
        dom = (stats.get("domains") or {}).get(domain) or {}
        cstats = dom.get("stats") or {}
        if cstats:
            desc_lines.append(f"Chunks: {cstats.get('count', 0)}")
            desc_lines.append(f"Avg chunk tokens: {cstats.get('mean', 0)}")
    desc_lines.append(f"Tokenizer: {TOKENIZER_ID}")
    return "\n".join(desc_lines)


def _call_llm(messages: List[Dict[str, str]], *, model: str, base_url: str, api_key: str, temperature: float, top_p: float, max_tokens: int) -> str:
    return chat_completion(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"top_p": top_p},
    )


def _validate_structure(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    users = payload.get("users")
    if not isinstance(users, list) or len(users) != 5:
        raise ValueError("users must be a list of length 5")
    for user in users:
        tasks = user.get("tasks") if isinstance(user, dict) else None
        if not isinstance(tasks, list) or len(tasks) != 5:
            raise ValueError("each user must have 5 tasks")
        for task in tasks:
            questions = task.get("questions") if isinstance(task, dict) else None
            if not isinstance(questions, list) or len(questions) != 5:
                raise ValueError("each task must have 5 questions")
    return users


def _flatten_questions(domain: str, users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    idx = 1
    for ui, user in enumerate(users):
        tasks = user.get("tasks") or []
        for ti, task in enumerate(tasks):
            questions = task.get("questions") or []
            for qi, q in enumerate(questions):
                if isinstance(q, dict):
                    q_text = q.get("question") or q.get("text") or ""
                else:
                    q_text = str(q)
                q_text = q_text.strip()
                if not q_text:
                    q_text = ""
                flat.append(
                    {
                        "question_id": f"{domain}-{idx:03d}",
                        "question": q_text,
                        "user_idx": ui,
                        "task_idx": ti,
                        "question_idx": qi,
                    }
                )
                idx += 1
    return flat


def _validate_questions(
    questions: List[Dict[str, Any]],
    dataset_desc: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
) -> Tuple[List[str], float]:
    labels: List[str] = []
    batch_size = 25
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        block = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(batch)])
        user_prompt = VALIDATOR_USER_TEMPLATE.format(dataset_desc=dataset_desc, questions_block=block)
        content = _call_llm(
            [
                {"role": "system", "content": VALIDATOR_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
        )
        obj = extract_json_object(content)
        batch_labels = obj.get("labels") if isinstance(obj, dict) else None
        if not isinstance(batch_labels, list) or len(batch_labels) != len(batch):
            raise ValueError("validator returned invalid labels")
        labels.extend([str(label).strip().lower() for label in batch_labels])
    pass_count = sum(1 for label in labels if label == "multi-doc")
    pass_rate = pass_count / max(1, len(labels))
    return labels, pass_rate


def _rewrite_failed(
    failed: List[Dict[str, Any]],
    dataset_desc: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
) -> List[str]:
    if not failed:
        return []
    block = "\n".join([f"{i+1}. {q['question']}" for i, q in enumerate(failed)])
    user_prompt = REWRITE_USER_TEMPLATE.format(dataset_desc=dataset_desc, questions_block=block)
    content = _call_llm(
        [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
        top_p=1.0,
        max_tokens=2048,
    )
    obj = extract_json_object(content)
    rewritten = obj.get("questions") if isinstance(obj, dict) else None
    if not isinstance(rewritten, list) or len(rewritten) != len(failed):
        raise ValueError("rewrite output invalid")
    return [str(q).strip() for q in rewritten]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UltraDomain questions for Mix/Legal.")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--base_url", default="https://api.deepseek.com/v1")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    args = parser.parse_args()

    ensure_dirs()
    api_key = get_api_key(args.api_key_env)
    domains = list(DOMAIN_LABELS.keys()) if args.domain == "all" else [args.domain]

    manifest: Dict[str, Any] = {
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": TOKENIZER_ID,
        "generated_at": now_iso(),
        "generation_params": {"temperature": 0.7, "top_p": 1.0, "max_output_tokens": 4096},
        "validator_params": {"temperature": 0.0, "top_p": 1.0},
        "domains": {},
    }

    for domain in domains:
        if domain not in DOMAIN_LABELS:
            continue
        dataset_desc = _build_dataset_desc(domain)
        user_prompt = USER_PROMPT_TEMPLATE.format(dataset_desc=dataset_desc, summary="")
        prompt_hash = sha256_text(SYSTEM_PROMPT + "\n" + user_prompt)
        content = _call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            temperature=0.7,
            top_p=1.0,
            max_tokens=4096,
        )
        payload = extract_json_object(content)
        users = _validate_structure(payload)
        flat = _flatten_questions(domain, users)
        labels, pass_rate = _validate_questions(flat, dataset_desc, model=args.model, base_url=args.base_url, api_key=api_key)
        regen_attempted = False
        if pass_rate < 0.95:
            regen_attempted = True
            failed = [q for q, label in zip(flat, labels) if label != "multi-doc"]
            rewritten = _rewrite_failed(failed, dataset_desc, model=args.model, base_url=args.base_url, api_key=api_key)
            for q, new_text in zip(failed, rewritten):
                q["question"] = new_text
                # update nested structure
                ui, ti, qi = q["user_idx"], q["task_idx"], q["question_idx"]
                users[ui]["tasks"][ti]["questions"][qi] = new_text
            labels, pass_rate = _validate_questions(flat, dataset_desc, model=args.model, base_url=args.base_url, api_key=api_key)

        out_payload = {
            "domain": domain,
            "generated_at": now_iso(),
            "prompt_hash": prompt_hash,
            "users": users,
            "questions": flat,
            "quality_gate": {
                "pass_rate": round(pass_rate, 4),
                "regen_attempted": regen_attempted,
            },
        }
        out_path = QUESTIONS_DIR / f"questions_{domain}.json"
        write_json(out_path, out_payload)
        manifest["domains"][domain] = {
            "questions_path": str(out_path),
            "prompt_hash": prompt_hash,
            "pass_rate": round(pass_rate, 4),
            "regen_attempted": regen_attempted,
        }

    write_json(QUESTIONS_DIR / "questions_manifest.json", manifest)
    print(f"Wrote questions to {QUESTIONS_DIR}")


if __name__ == "__main__":
    main()
