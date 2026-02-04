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
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize UltraDomain run metadata.")
    parser.add_argument("--dataset_id", default="TBD_ULTRADOMAIN_DATASET_ID")
    parser.add_argument("--split", default="train")
    parser.add_argument("--base_url", default="https://api.deepseek.com/v1")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--protocol_version", default="v1")
    args = parser.parse_args()

    ensure_dirs()
    cfg = config_loader.load_config()
    embed_cfg = (cfg.get("retriever") or {}).get("embedding") or {}

    system_configs: Dict[str, Any] = {
        "retrieval_budget_tokens": 12000,
        "prefilter_top_k": 40,
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
            "temperature": 0.2,
            "top_p": 1.0,
            "max_output_tokens": 1024,
            "context_length": 128000,
        },
        "judge_llm": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": 0.0,
            "top_p": 1.0,
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
