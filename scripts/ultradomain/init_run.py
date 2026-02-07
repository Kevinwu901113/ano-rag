from __future__ import annotations

import argparse
from typing import Any, Dict

from relrag.config.config_loader import config as config_loader
from scripts.ultradomain.common import (
    INDEX_DIR,
    OUTPUT_ROOT,
    RUN_META_DIR,
    TOKENIZER_ID,
    ensure_dirs,
    get_git_commit,
    now_iso,
    ultradomain_get,
    write_json,
)


def main() -> None:
    dataset_id_default = ultradomain_get("dataset.dataset_id", "TBD_ULTRADOMAIN_DATASET_ID")
    split_default = ultradomain_get("dataset.split", "train")
    base_url_default = ultradomain_get("llm.base_url", "https://api.deepseek.com/v1")
    model_default = ultradomain_get("llm.model", "deepseek-chat")
    protocol_version_default = ultradomain_get("protocol.version", "v1")
    retrieval_budget_default = int(ultradomain_get("retrieval.budget_tokens", 12000) or 12000)
    prefilter_top_k_default = int(ultradomain_get("retrieval.prefilter_top_k", 40) or 40)
    answer_temp_default = float(ultradomain_get("answer.temperature", 0.2) or 0.2)
    answer_top_p_default = float(ultradomain_get("answer.top_p", 1.0) or 1.0)
    answer_max_tokens_default = int(ultradomain_get("answer.max_output_tokens", 1024) or 1024)
    judge_temp_default = float(ultradomain_get("judge.temperature", 0.0) or 0.0)
    judge_top_p_default = float(ultradomain_get("judge.top_p", 1.0) or 1.0)

    parser = argparse.ArgumentParser(description="Initialize UltraDomain run metadata.")
    parser.add_argument(
        "--dataset_id",
        default=dataset_id_default,
        help="HuggingFace dataset id for UltraDomain (must be set explicitly).",
    )
    parser.add_argument("--split", default=split_default)
    parser.add_argument("--base_url", default=base_url_default)
    parser.add_argument("--model", default=model_default)
    parser.add_argument("--protocol_version", default=protocol_version_default)
    args = parser.parse_args()

    if str(args.dataset_id).startswith("TBD_") or "TBD" in str(args.dataset_id):
        raise SystemExit("Please provide a real --dataset_id (placeholder TBD_* is not allowed).")

    ensure_dirs()
    cfg = config_loader.load_config()
    embed_cfg = (cfg.get("retriever") or {}).get("embedding") or {}

    system_configs: Dict[str, Any] = {
        "retrieval_budget_tokens": retrieval_budget_default,
        "prefilter_top_k": prefilter_top_k_default,
        "reranker": {"relrag_full": True, "others": False},
        "systems": {
            "RelRAG-full": {"structured": True, "hybrid": True, "bm25": True, "dense": True, "reranker": True},
            "BM25-only": {"structured": False, "hybrid": False, "bm25": True, "dense": False, "reranker": False},
            "Dense-only": {"structured": False, "hybrid": False, "bm25": False, "dense": True, "reranker": False},
            "Hybrid-only": {"structured": False, "hybrid": False, "bm25": True, "dense": True, "reranker": False},
        },
    }

    run_config: Dict[str, Any] = {
        "protocol_version": args.protocol_version,
        "dataset_id": args.dataset_id,
        "split": args.split,
        "output_root": str(OUTPUT_ROOT),
        "answer_llm": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": answer_temp_default,
            "top_p": answer_top_p_default,
            "max_output_tokens": answer_max_tokens_default,
            "context_length": 128000,
        },
        "judge_llm": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": judge_temp_default,
            "top_p": judge_top_p_default,
        },
        "tokenizer": TOKENIZER_ID,
        "embedding": {
            "provider": embed_cfg.get("provider"),
            "model": embed_cfg.get("model"),
            "endpoint": embed_cfg.get("endpoint"),
        },
        "git_commit": get_git_commit(),
        "generated_at": now_iso(),
    }

    write_json(RUN_META_DIR / "system_configs.json", system_configs)
    write_json(RUN_META_DIR / "run_config.json", run_config)
    (RUN_META_DIR / "protocol_version").write_text(args.protocol_version, encoding="utf-8")
    print(f"Initialized run metadata in {RUN_META_DIR}")


if __name__ == "__main__":
    main()
