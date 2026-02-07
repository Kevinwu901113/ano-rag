from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List

from scripts.ultradomain.common import RUN_META_DIR, ensure_dirs, get_git_commit, now_iso, ultradomain_get, write_json


def _append_opt(args: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    args.extend([flag, text])


def _run_module(step: str, module: str, module_args: List[str], audit: List[Dict[str, Any]]) -> None:
    cmd = [sys.executable, "-m", module, *module_args]
    started_ts = time.time()
    started_at = now_iso()
    entry: Dict[str, Any] = {
        "step": step,
        "module": module,
        "command": " ".join(shlex.quote(x) for x in cmd),
        "started_at": started_at,
        "status": "running",
    }
    audit.append(entry)
    _write_audit(audit)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        entry["status"] = "failed"
        entry["exit_code"] = exc.returncode
        entry["finished_at"] = now_iso()
        entry["duration_sec"] = round(time.time() - started_ts, 3)
        _write_audit(audit)
        raise
    entry["status"] = "ok"
    entry["finished_at"] = now_iso()
    entry["duration_sec"] = round(time.time() - started_ts, 3)
    _write_audit(audit)


def _write_audit(steps: List[Dict[str, Any]]) -> None:
    ensure_dirs()
    write_json(
        RUN_META_DIR / "pipeline_run.json",
        {
            "generated_at": now_iso(),
            "git_commit": get_git_commit(),
            "steps": steps,
        },
    )


def main() -> None:
    dataset_id_default = ultradomain_get("dataset.dataset_id", "TBD_ULTRADOMAIN_DATASET_ID")
    split_default = ultradomain_get("dataset.split", "train")
    config_name_default = ultradomain_get("dataset.config_name", None)
    cache_dir_default = ultradomain_get("dataset.cache_dir", None)
    limit_default = int(ultradomain_get("dataset.limit", 0) or 0)
    domain_default = ultradomain_get("dataset.domain", "all")

    protocol_version_default = ultradomain_get("protocol.version", "v1")
    output_root_default = ultradomain_get(
        "protocol.output_root",
        "result/ultradomain/lightrag_protocol_v1_mix_legal",
    )

    base_url_default = ultradomain_get("llm.base_url", "https://api.deepseek.com/v1")
    model_default = ultradomain_get("llm.model", "deepseek-chat")
    api_key_env_default = ultradomain_get("llm.api_key_env", "DEEPSEEK_API_KEY")

    answer_temp_default = float(ultradomain_get("answer.temperature", 0.2) or 0.2)
    answer_top_p_default = float(ultradomain_get("answer.top_p", 1.0) or 1.0)
    answer_max_tokens_default = int(ultradomain_get("answer.max_output_tokens", 1024) or 1024)
    budget_tokens_default = int(ultradomain_get("retrieval.budget_tokens", 12000) or 12000)
    prefilter_top_k_default = int(ultradomain_get("retrieval.prefilter_top_k", 40) or 40)

    resume_default = bool(ultradomain_get("pipeline.resume", False))
    skip_validate_default = bool(ultradomain_get("pipeline.skip_validate", False))
    validate_strict_default = bool(ultradomain_get("validation.strict", True))
    validate_expected_q_default = int(ultradomain_get("validation.expected_questions_per_domain", 125) or 125)

    skip_relrag_default = bool(ultradomain_get("indexing.skip_relrag", False))
    skip_chunk_default = bool(ultradomain_get("indexing.skip_chunk", False))
    index_llm_provider_default = ultradomain_get("indexing.llm_provider", "vllm")
    index_llm_endpoint_default = ultradomain_get("indexing.llm_endpoint", None)
    index_llm_model_default = ultradomain_get("indexing.llm_model", None)
    index_llm_api_key_default = ultradomain_get("indexing.llm_api_key", None)
    index_llm_temp_default = float(ultradomain_get("indexing.llm_temperature", 0.0) or 0.0)
    index_llm_max_tokens_default = ultradomain_get("indexing.llm_max_tokens", None)
    embed_provider_default = ultradomain_get("indexing.embed_provider", None)
    embed_model_default = ultradomain_get("indexing.embed_model", None)
    embed_endpoint_default = ultradomain_get("indexing.embed_endpoint", None)
    embed_api_key_default = ultradomain_get("indexing.embed_api_key", None)

    chunk_size_default = int(ultradomain_get("chunking.chunk_size", 1200) or 1200)
    chunk_overlap_default = int(ultradomain_get("chunking.overlap", 100) or 100)

    parser = argparse.ArgumentParser(description="Run the UltraDomain LightRAG-style protocol pipeline.")
    parser.add_argument("--dataset_id", default=dataset_id_default, help="UltraDomain dataset id on HuggingFace.")
    parser.add_argument("--split", default=split_default)
    parser.add_argument("--config_name", default=config_name_default)
    parser.add_argument("--cache_dir", default=cache_dir_default)
    parser.add_argument("--limit", type=int, default=limit_default)
    parser.add_argument("--domain", default=domain_default)
    parser.add_argument("--protocol_version", default=protocol_version_default)

    parser.add_argument("--base_url", default=base_url_default)
    parser.add_argument("--model", default=model_default)
    parser.add_argument("--api_key_env", default=api_key_env_default)

    parser.add_argument("--answer_temperature", type=float, default=answer_temp_default)
    parser.add_argument("--answer_top_p", type=float, default=answer_top_p_default)
    parser.add_argument("--answer_max_output_tokens", type=int, default=answer_max_tokens_default)
    parser.add_argument("--budget_tokens", type=int, default=budget_tokens_default)
    parser.add_argument("--prefilter_top_k", type=int, default=prefilter_top_k_default)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=resume_default)

    parser.add_argument("--chunk_size", type=int, default=chunk_size_default)
    parser.add_argument("--chunk_overlap", type=int, default=chunk_overlap_default)

    parser.add_argument("--skip_relrag", action=argparse.BooleanOptionalAction, default=skip_relrag_default)
    parser.add_argument("--skip_chunk", action=argparse.BooleanOptionalAction, default=skip_chunk_default)
    parser.add_argument("--index_llm_provider", default=index_llm_provider_default, choices=["vllm", "openai"])
    parser.add_argument("--index_llm_endpoint", default=index_llm_endpoint_default)
    parser.add_argument("--index_llm_model", default=index_llm_model_default)
    parser.add_argument("--index_llm_api_key", default=index_llm_api_key_default)
    parser.add_argument("--index_llm_temperature", type=float, default=index_llm_temp_default)
    parser.add_argument("--index_llm_max_tokens", type=int, default=index_llm_max_tokens_default)

    parser.add_argument("--embed_provider", default=embed_provider_default)
    parser.add_argument("--embed_model", default=embed_model_default)
    parser.add_argument("--embed_endpoint", default=embed_endpoint_default)
    parser.add_argument("--embed_api_key", default=embed_api_key_default)

    parser.add_argument("--skip_validate", action=argparse.BooleanOptionalAction, default=skip_validate_default)
    parser.add_argument(
        "--validate_strict",
        action=argparse.BooleanOptionalAction,
        default=validate_strict_default,
        help="Whether validation should fail the pipeline on errors.",
    )
    parser.add_argument(
        "--expected_questions_per_domain",
        type=int,
        default=validate_expected_q_default,
        help="Expected frozen question count for each domain in validation.",
    )
    args = parser.parse_args()

    if str(args.dataset_id).startswith("TBD_") or "TBD" in str(args.dataset_id):
        raise SystemExit("Please set ultradomain.dataset.dataset_id in config.yaml or pass --dataset_id.")

    ensure_dirs()
    audit_steps: List[Dict[str, Any]] = []

    init_args = [
        "--dataset_id",
        args.dataset_id,
        "--split",
        args.split,
        "--base_url",
        args.base_url,
        "--model",
        args.model,
        "--protocol_version",
        args.protocol_version,
    ]
    _run_module("init_run", "scripts.ultradomain.init_run", init_args, audit_steps)

    prepare_args = [
        "--dataset_id",
        args.dataset_id,
        "--split",
        args.split,
        "--domain",
        args.domain,
    ]
    _append_opt(prepare_args, "--config_name", args.config_name)
    _append_opt(prepare_args, "--cache_dir", args.cache_dir)
    if args.limit > 0:
        prepare_args.extend(["--limit", str(args.limit)])
    _run_module("prepare_docs", "scripts.ultradomain.prepare_docs", prepare_args, audit_steps)

    build_chunks_args = [
        "--domain",
        args.domain,
        "--chunk_size",
        str(args.chunk_size),
        "--overlap",
        str(args.chunk_overlap),
    ]
    _run_module("build_chunks", "scripts.ultradomain.build_chunks", build_chunks_args, audit_steps)

    build_indices_args = [
        "--domain",
        args.domain,
        "--llm_provider",
        args.index_llm_provider,
        "--llm_temperature",
        str(args.index_llm_temperature),
    ]
    build_indices_args.append("--skip_relrag" if args.skip_relrag else "--no-skip_relrag")
    build_indices_args.append("--skip_chunk" if args.skip_chunk else "--no-skip_chunk")
    _append_opt(build_indices_args, "--llm_endpoint", args.index_llm_endpoint)
    _append_opt(build_indices_args, "--llm_model", args.index_llm_model)
    _append_opt(build_indices_args, "--llm_api_key", args.index_llm_api_key)
    _append_opt(build_indices_args, "--llm_max_tokens", args.index_llm_max_tokens)
    _append_opt(build_indices_args, "--embed_provider", args.embed_provider)
    _append_opt(build_indices_args, "--embed_model", args.embed_model)
    _append_opt(build_indices_args, "--embed_endpoint", args.embed_endpoint)
    _append_opt(build_indices_args, "--embed_api_key", args.embed_api_key)
    _run_module("build_indices", "scripts.ultradomain.build_indices", build_indices_args, audit_steps)

    qgen_args = [
        "--domain",
        args.domain,
        "--base_url",
        args.base_url,
        "--model",
        args.model,
        "--api_key_env",
        args.api_key_env,
    ]
    _run_module("generate_questions", "scripts.ultradomain.generate_questions", qgen_args, audit_steps)

    answer_args = [
        "--domain",
        args.domain,
        "--system",
        "all",
        "--base_url",
        args.base_url,
        "--model",
        args.model,
        "--api_key_env",
        args.api_key_env,
        "--temperature",
        str(args.answer_temperature),
        "--top_p",
        str(args.answer_top_p),
        "--max_output_tokens",
        str(args.answer_max_output_tokens),
        "--budget_tokens",
        str(args.budget_tokens),
        "--prefilter_top_k",
        str(args.prefilter_top_k),
    ]
    answer_args.append("--resume" if args.resume else "--no-resume")
    _run_module("run_answers", "scripts.ultradomain.run_answers", answer_args, audit_steps)

    judge_args = [
        "--domain",
        args.domain,
        "--base_url",
        args.base_url,
        "--model",
        args.model,
        "--api_key_env",
        args.api_key_env,
    ]
    judge_args.append("--resume" if args.resume else "--no-resume")
    _run_module("judge_pairwise", "scripts.ultradomain.judge_pairwise", judge_args, audit_steps)

    if not args.skip_validate:
        validate_args = [
            "--output_root",
            str(output_root_default),
            "--protocol_version",
            args.protocol_version,
            "--expected_questions_per_domain",
            str(args.expected_questions_per_domain),
        ]
        validate_args.append("--strict" if args.validate_strict else "--no-strict")
        _run_module("summary_validate", "scripts.ultradomain.validate_protocol_outputs", validate_args, audit_steps)

    print("UltraDomain pipeline completed.")
    print(f"Audit file: {RUN_META_DIR / 'pipeline_run.json'}")


if __name__ == "__main__":
    main()
