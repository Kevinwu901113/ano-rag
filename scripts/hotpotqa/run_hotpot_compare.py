#!/usr/bin/env python3
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer
import yaml


def load_hotpot_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    return records


def write_official_gold(records: List[Dict[str, Any]], out_path: Path) -> None:
    gold: List[Dict[str, Any]] = []
    for rec in records:
        qid = rec.get("_id") or rec.get("id")
        if not qid:
            continue
        sp = rec.get("supporting_facts") or []
        normalized_sp: List[List[Any]] = []
        for item in sp:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    normalized_sp.append([str(item[0]), int(item[1])])
                except (TypeError, ValueError):
                    continue
        gold.append({"_id": str(qid), "answer": rec.get("answer", ""), "supporting_facts": normalized_sp})
    out_path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")


def write_official_pred(preds: Dict[str, str], out_path: Path) -> None:
    payload = {"answer": {}, "sp": {}}
    for qid, ans in preds.items():
        payload["answer"][qid] = ans
        payload["sp"][qid] = []
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_official(pred_path: Path, gold_path: Path, metrics_path: Path, repo_root: Path) -> None:
    import importlib.util

    eval_path = repo_root / "eval" / "hotpot_evaluate_v1.py"
    spec = importlib.util.spec_from_file_location("hotpot_eval", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load evaluator from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    metrics = module.eval(str(pred_path), str(gold_path))
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


class TokenBudgeter:
    def __init__(self, tokenizer_model: str, max_context_len: int, safety_margin_tokens: int):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model, trust_remote_code=True, use_fast=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model, trust_remote_code=True, use_fast=False
            )
        self.max_context_len = max_context_len
        self.safety_margin_tokens = safety_margin_tokens

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        if has_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts)

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        prompt = self._messages_to_prompt(messages)
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

    def available_output_tokens(self, messages: List[Dict[str, str]], requested: int) -> int:
        prompt_tokens = self.count_message_tokens(messages)
        available = self.max_context_len - self.safety_margin_tokens - prompt_tokens
        return max(0, min(requested, available))


class VLLMChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        tokenizer_model: str,
        max_context_len: int,
        safety_margin_tokens: int,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.budgeter = TokenBudgeter(
            tokenizer_model=tokenizer_model,
            max_context_len=max_context_len,
            safety_margin_tokens=safety_margin_tokens,
        )

    def chat(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        max_tokens = self.budgeter.available_output_tokens(messages, max_tokens)
        if max_tokens <= 0:
            return ""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        available = self.budgeter.available_output_tokens(messages, max_tokens)
        overhead_tokens = self.budgeter.count_message_tokens([{"role": "user", "content": ""}])
        allowed_prompt_tokens = max(0, self.budgeter.max_context_len - self.budgeter.safety_margin_tokens - max_tokens - overhead_tokens)
        return self.budgeter.truncate_text_to_tokens(prompt, allowed_prompt_tokens)

    def truncate_user_prompt(
        self,
        user_prompt: str,
        max_tokens: int,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        # Keep user prompt within budget after accounting for system/history messages.
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": ""})
        overhead_tokens = self.budgeter.count_message_tokens(messages)
        allowed_prompt_tokens = max(
            0,
            self.budgeter.max_context_len
            - self.budgeter.safety_margin_tokens
            - max_tokens
            - overhead_tokens,
        )
        return self.budgeter.truncate_text_to_tokens(user_prompt, allowed_prompt_tokens)


def build_context_text(context: List[List[Any]]) -> str:
    parts = []
    for item in context:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = item[0]
        sents = item[1]
        if isinstance(title, str) and isinstance(sents, list):
            parts.append(f"### {title}\n{' '.join(sents)}")
    return "\n\n".join(parts)


LIGHTRAG_TUPLE_DELIMITER = "<|#|>"
LIGHTRAG_COMPLETION_DELIMITER = "<|COMPLETE|>"


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _normalize_extraction_field(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip()
    cleaned = cleaned.strip("`").strip().strip('"').strip("'").strip()
    cleaned = re.sub(r"<+$", "", cleaned).strip()
    return cleaned


def _repair_lightrag_extraction_output(
    raw_text: str,
    tuple_delimiter: str = LIGHTRAG_TUPLE_DELIMITER,
    completion_delimiter: str = LIGHTRAG_COMPLETION_DELIMITER,
) -> str:
    text = _strip_markdown_fence(str(raw_text or ""))
    if not text:
        return completion_delimiter

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?i)relationship(?=\s*<\|#\|>)", "relation", text)
    text = text.replace(completion_delimiter.lower(), completion_delimiter)
    text = text.replace(completion_delimiter, f"\n{completion_delimiter}\n")

    raw_records: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line == completion_delimiter:
            continue
        parts = re.split(
            r"(?=(?:entity|relation|relationship)\s*<\|#\|>)",
            line,
            flags=re.IGNORECASE,
        )
        for part in parts:
            part = part.strip()
            if not part:
                continue
            part = re.sub(r"^[\-\*\d\.\)\(\s]+", "", part).strip()
            match = re.search(
                r"(entity|relation|relationship)\s*<\|#\|>",
                part,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            normalized = part[match.start() :].strip()
            if tuple_delimiter not in normalized:
                continue
            normalized = re.sub(
                r"(?i)^relationship(?=\s*<\|#\|>)",
                "relation",
                normalized,
            )
            raw_records.append(normalized)

    repaired_records: List[str] = []
    for record in raw_records:
        attrs = [item.strip() for item in record.split(tuple_delimiter)]
        if not attrs:
            continue
        kind = attrs[0].strip().lower()
        if "entity" in kind:
            fields = ["entity"] + attrs[1:]
            fields = [_normalize_extraction_field(v) for v in fields]
            if len(fields) < 4:
                fields += [""] * (4 - len(fields))
            elif len(fields) > 4:
                merged_desc = "; ".join([x for x in fields[3:] if x])
                fields = fields[:3] + [merged_desc]
            name = fields[1] or "Unknown Entity"
            entity_type = fields[2] or "other"
            desc = fields[3] or f"{name} is an entity mentioned in the text."
            repaired_records.append(
                tuple_delimiter.join(["entity", name, entity_type, desc])
            )
            continue
        if "relation" in kind:
            fields = ["relation"] + attrs[1:]
            fields = [_normalize_extraction_field(v) for v in fields]
            if len(fields) < 5:
                fields += [""] * (5 - len(fields))
            elif len(fields) > 5:
                merged_desc = "; ".join([x for x in fields[4:] if x])
                fields = fields[:4] + [merged_desc]
            src = fields[1] or "Unknown Source"
            tgt = fields[2] or "Unknown Target"
            keywords = fields[3] or "related"
            desc = fields[4] or f"{src} is related to {tgt}."
            repaired_records.append(
                tuple_delimiter.join(["relation", src, tgt, keywords, desc])
            )

    if not repaired_records:
        repaired_text = text.strip()
        if completion_delimiter not in repaired_text:
            repaired_text = f"{repaired_text}\n{completion_delimiter}".strip()
        return repaired_text

    return "\n".join(repaired_records + [completion_delimiter]).strip()


def _tail_text(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return "(log file missing)"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return "(empty log)"
        return "\n".join(lines[-max_lines:])
    except Exception as exc:  # pragma: no cover - defensive
        return f"(failed to read log: {exc})"


def run_raptor(args: argparse.Namespace, records: List[Dict[str, Any]], out_dir: Path) -> Path:
    pred_path = out_dir / "raptor_pred.json"
    gold_path = out_dir / "raptor_gold.json"
    workers = max(
        1,
        int(getattr(args, "raptor_workers", getattr(args, "question_workers", 1))),
    )
    thread_cap = max(1, int(getattr(args, "raptor_threads", 1)))
    raptor_env = os.environ.copy()
    raptor_env["OPENAI_BASE_URL"] = str(args.llm_base_url)
    raptor_env["VLLM_MODEL"] = str(args.llm_model)
    raptor_env["OPENAI_API_KEY"] = str(args.api_key or "EMPTY")
    raptor_env["OPENAI_EMBED_BASE_URL"] = str(args.embed_base_url)
    raptor_env["TOKENIZERS_PARALLELISM"] = "false"
    raptor_env["PYTHONUNBUFFERED"] = "1"
    for var_name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        raptor_env[var_name] = str(thread_cap)

    base_cmd = [
        args.raptor_python or sys.executable,
        str(args.repo_root / "RAPTOR" / "raptor" / "run_hotpotqa_raptor_vllm.py"),
        "--embed_backend",
        "vllm",
        "--embed_model",
        args.embed_model,
        "--embed_base_url",
        args.embed_base_url,
        "--tokenizer_model",
        args.tokenizer_model,
        "--max_context_len",
        str(args.max_context_len),
        "--safety_margin_tokens",
        str(args.safety_margin_tokens),
        "--per_example_retries",
        str(max(0, int(getattr(args, "raptor_per_example_retries", 1)))),
    ]
    if bool(getattr(args, "raptor_fail_on_example_error", False)):
        base_cmd.append("--fail_on_example_error")

    if workers <= 1 or len(records) <= 1:
        cmd = base_cmd + [
            "--data_jsonl",
            str(args.data),
            "--limit",
            str(args.limit),
            "--out_pred",
            str(pred_path),
            "--out_gold",
            str(gold_path),
        ]
        result = subprocess.run(
            cmd,
            check=False,
            env=raptor_env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stdout_tail = "\n".join((result.stdout or "").splitlines()[-80:])
            stderr_tail = "\n".join((result.stderr or "").splitlines()[-80:])
            raise RuntimeError(
                "RAPTOR run failed.\n"
                f"cmd: {' '.join(cmd)}\n"
                f"returncode: {result.returncode}\n"
                f"stdout tail:\n{stdout_tail}\n\nstderr tail:\n{stderr_tail}"
            )
        return pred_path

    shard_dir = out_dir / "_raptor_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    worker_count = min(workers, len(records))
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(worker_count)]
    for idx, rec in enumerate(records):
        shards[idx % worker_count].append(rec)

    procs: List[subprocess.Popen] = []
    shard_log_paths: List[Path] = []
    shard_log_handles = []
    shard_pred_paths: List[Path] = []
    shard_gold_paths: List[Path] = []

    for shard_idx, shard_records in enumerate(shards):
        shard_data_path = shard_dir / f"data_{shard_idx}.jsonl"
        shard_pred_path = shard_dir / f"pred_{shard_idx}.json"
        shard_gold_path = shard_dir / f"gold_{shard_idx}.json"
        with shard_data_path.open("w", encoding="utf-8") as handle:
            for rec in shard_records:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

        cmd = base_cmd + [
            "--data_jsonl",
            str(shard_data_path),
            "--limit",
            "0",
            "--out_pred",
            str(shard_pred_path),
            "--out_gold",
            str(shard_gold_path),
        ]
        shard_log_path = shard_dir / f"shard_{shard_idx}.log"
        log_handle = shard_log_path.open("w", encoding="utf-8")
        procs.append(
            subprocess.Popen(
                cmd,
                env=raptor_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        )
        shard_log_handles.append(log_handle)
        shard_log_paths.append(shard_log_path)
        shard_pred_paths.append(shard_pred_path)
        shard_gold_paths.append(shard_gold_path)

    exit_codes = [proc.wait() for proc in procs]
    for handle in shard_log_handles:
        handle.close()
    if any(code != 0 for code in exit_codes):
        details = []
        for idx, code in enumerate(exit_codes):
            if code == 0:
                continue
            tail = _tail_text(shard_log_paths[idx], max_lines=120)
            details.append(
                f"[shard={idx} exit={code} log={shard_log_paths[idx]}]\n{tail}"
            )
        detail_text = "\n\n".join(details)
        raise RuntimeError(
            f"RAPTOR shard parallel run failed: {exit_codes}\n{detail_text}"
        )

    merged_pred: Dict[str, Dict[str, Any]] = {"answer": {}, "sp": {}}
    merged_gold: List[Dict[str, Any]] = []
    for shard_pred_path, shard_gold_path in zip(shard_pred_paths, shard_gold_paths):
        shard_pred_payload = json.loads(shard_pred_path.read_text(encoding="utf-8"))
        shard_gold_payload = json.loads(shard_gold_path.read_text(encoding="utf-8"))
        merged_pred["answer"].update(shard_pred_payload.get("answer", {}))
        merged_pred["sp"].update(shard_pred_payload.get("sp", {}))
        if isinstance(shard_gold_payload, list):
            merged_gold.extend(shard_gold_payload)

    pred_path.write_text(json.dumps(merged_pred, ensure_ascii=False, indent=2), encoding="utf-8")
    gold_path.write_text(json.dumps(merged_gold, ensure_ascii=False, indent=2), encoding="utf-8")
    return pred_path


def _resolve_graphrag_cli(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        cli = Path(explicit_path).expanduser()
        if cli.exists():
            return str(cli)
        raise RuntimeError(f"Configured graphrag CLI path does not exist: {cli}")

    detected = shutil.which("graphrag")
    if detected:
        return detected

    env_cli = Path(sys.executable).resolve().parent / "graphrag"
    if env_cli.exists():
        return str(env_cli)

    raise RuntimeError(
        "graphrag CLI not found. Install with `pip install graphrag` or set --graphrag_cli."
    )


def _ensure_graphrag_workspace(
    root: Path, llm_model: str, embed_model: str, graphrag_cli: str
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    settings_path = root / "settings.yaml"
    env_path = root / ".env"
    if not settings_path.exists():
        subprocess.run(
            [graphrag_cli, "init", "--root", str(root), "--model", llm_model, "--embedding", embed_model],
            check=True,
        )
    if not env_path.exists():
        env_path.write_text("GRAPHRAG_API_KEY=EMPTY\n", encoding="utf-8")
    return settings_path


def _patch_graphrag_settings(
    settings_path: Path,
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    embed_encoding_format: str,
    embed_vector_size: Optional[int],
    max_tokens: int,
    temperature: float,
    relaxed_pruning: bool,
    max_context_tokens: int,
) -> None:
    import yaml

    cfg = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}

    models = cfg.setdefault("models", {})
    if isinstance(models, dict):
        models.setdefault("vllm_chat", {})
        models.setdefault("vllm_embed", {})
        models["vllm_chat"].update(
            {
                "type": "openai_chat",
                "model": llm_model,
                "api_base": llm_base_url,
                "api_key": "${GRAPHRAG_API_KEY}",
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }
        )
        models["vllm_embed"].update(
            {
                "type": "openai_embedding",
                "model": embed_model,
                "api_base": embed_base_url,
                "api_key": "${GRAPHRAG_API_KEY}",
                "encoding_format": embed_encoding_format,
            }
        )
        cfg["default_chat_model"] = "vllm_chat"
        cfg["default_embedding_model"] = "vllm_embed"

    completion_models = cfg.get("completion_models")
    if isinstance(completion_models, dict) and completion_models:
        target_id = "default_completion_model" if "default_completion_model" in completion_models else next(iter(completion_models))
        model_cfg = completion_models.get(target_id)
        if isinstance(model_cfg, dict):
            model_cfg.update(
                {
                    "model_provider": "openai",
                    "model": llm_model,
                    "api_base": llm_base_url,
                    "api_key": "${GRAPHRAG_API_KEY}",
                    "auth_method": "api_key",
                    "call_args": {
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                    },
                }
            )
            completion_models[target_id] = model_cfg

    embedding_models = cfg.get("embedding_models")
    if isinstance(embedding_models, dict) and embedding_models:
        target_id = "default_embedding_model" if "default_embedding_model" in embedding_models else next(iter(embedding_models))
        model_cfg = embedding_models.get(target_id)
        if isinstance(model_cfg, dict):
            call_args = model_cfg.get("call_args")
            if not isinstance(call_args, dict):
                call_args = {}
            call_args["encoding_format"] = embed_encoding_format
            model_cfg.update(
                {
                    "model_provider": "openai",
                    "model": embed_model,
                    "api_base": embed_base_url,
                    "api_key": "${GRAPHRAG_API_KEY}",
                    "auth_method": "api_key",
                    "call_args": call_args,
                }
            )
            embedding_models[target_id] = model_cfg

    if isinstance(cfg.get("llm"), dict):
        cfg["llm"].update(
            {
                "type": "openai_chat",
                "model": llm_model,
                "api_base": llm_base_url,
                "api_key": "${GRAPHRAG_API_KEY}",
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }
        )
    if isinstance(cfg.get("embeddings"), dict):
        cfg["embeddings"].update(
            {
                "type": "openai_embedding",
                "model": embed_model,
                "api_base": embed_base_url,
                "api_key": "${GRAPHRAG_API_KEY}",
                "encoding_format": embed_encoding_format,
            }
        )

    if isinstance(cfg.get("local_search"), dict):
        cfg["local_search"]["max_context_tokens"] = int(max_context_tokens)
    if isinstance(cfg.get("global_search"), dict):
        cfg["global_search"]["max_context_tokens"] = int(max_context_tokens)
        cfg["global_search"]["data_max_tokens"] = int(max_context_tokens)
    if isinstance(cfg.get("drift_search"), dict):
        cfg["drift_search"]["data_max_tokens"] = int(max_context_tokens)
        cfg["drift_search"]["local_search_max_data_tokens"] = int(max_context_tokens)
        cfg["drift_search"]["primer_llm_max_tokens"] = int(max_context_tokens)
    if isinstance(cfg.get("basic_search"), dict):
        cfg["basic_search"]["max_context_tokens"] = int(max_context_tokens)

    vector_store = cfg.get("vector_store")
    if isinstance(vector_store, dict) and embed_vector_size is not None:
        index_schema = vector_store.get("index_schema")
        if not isinstance(index_schema, dict):
            index_schema = {}
        for index_name in (
            "text_unit_text",
            "community_full_content",
            "entity_description",
        ):
            schema = index_schema.get(index_name)
            if not isinstance(schema, dict):
                schema = {}
            schema.setdefault("index_name", index_name)
            schema.setdefault("id_field", "id")
            schema.setdefault("vector_field", "vector")
            schema["vector_size"] = int(embed_vector_size)
            index_schema[index_name] = schema
        vector_store["index_schema"] = index_schema

    if relaxed_pruning:
        prune_cfg = cfg.setdefault("prune_graph", {})
        if isinstance(prune_cfg, dict):
            prune_cfg["min_node_freq"] = 1
            prune_cfg["min_node_degree"] = 0
            prune_cfg["min_edge_weight_pct"] = 0.0
            prune_cfg["remove_ego_nodes"] = False

    cfg.setdefault("encoding_model", "cl100k_base")
    settings_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _probe_embedding_vector_size(
    embed_base_url: str,
    api_key: str,
    embed_model: str,
    embed_encoding_format: str,
) -> int:
    import base64

    client = OpenAI(base_url=embed_base_url, api_key=api_key)
    response = client.embeddings.create(
        input=["probe"],
        model=embed_model,
        encoding_format=embed_encoding_format,
    )
    vector = response.data[0].embedding
    if isinstance(vector, list):
        return len(vector)
    if isinstance(vector, str):
        raw = base64.b64decode(vector.encode("utf-8"), validate=False)
        if len(raw) % 4 != 0:
            raise RuntimeError(
                "Unexpected base64 embedding payload length (not divisible by 4)."
            )
        return len(raw) // 4
    raise RuntimeError(f"Unsupported embedding payload type: {type(vector)!r}")


def run_graphrag(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    out_dir: Path,
    mode: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "graphrag_pred.json"
    workers = max(1, int(getattr(args, "question_workers", 1)))
    graphrag_llm_base_url = args.graphrag_llm_base_url or args.llm_base_url
    graphrag_llm_model = args.graphrag_llm_model or args.llm_model
    graphrag_cli = _resolve_graphrag_cli(getattr(args, "graphrag_cli", None))
    relaxed_pruning = bool(
        getattr(args, "graphrag_relaxed_pruning", str(args.mode).lower() == "strict")
    )
    max_context_tokens = max(
        1024,
        int(args.max_context_len)
        - int(args.safety_margin_tokens)
        - int(args.max_tokens)
        - 1024,
    )
    embed_vector_size = _probe_embedding_vector_size(
        embed_base_url=args.embed_base_url,
        api_key=args.api_key or "EMPTY",
        embed_model=args.embed_model,
        embed_encoding_format=args.embed_encoding_format,
    )
    graphrag_env = os.environ.copy()
    graphrag_env.setdefault("GRAPHRAG_API_KEY", args.api_key or "EMPTY")
    runtime_patch_dir = Path(__file__).resolve().parent / "graphrag_runtime_patch"
    existing_pythonpath = graphrag_env.get("PYTHONPATH")
    graphrag_env["PYTHONPATH"] = (
        f"{runtime_patch_dir}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(runtime_patch_dir)
    )
    graphrag_env["GRAPHRAG_SCHEMA_STRATEGY"] = str(args.graphrag_schema_strategy).lower()

    def _last_non_empty_line(text: str) -> str:
        for line in reversed(text.splitlines()):
            if line.strip():
                return line.strip()
        return ""

    def _tail(text: str, max_lines: int = 40) -> str:
        lines = [line for line in text.splitlines() if line.strip()]
        return "\n".join(lines[-max_lines:])

    def _run_graphrag_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=graphrag_env,
        )
        if check and result.returncode != 0:
            stdout_tail = _tail(result.stdout or "")
            stderr_tail = _tail(result.stderr or "")
            details = []
            if stdout_tail:
                details.append(f"stdout:\n{stdout_tail}")
            if stderr_tail:
                details.append(f"stderr:\n{stderr_tail}")
            detail_text = "\n\n".join(details) if details else "(no output captured)"
            raise RuntimeError(f"GraphRAG command failed ({result.returncode}): {' '.join(cmd)}\n{detail_text}")
        return result

    def _prepare_workspace(workspace: Path, content: str, filename: str) -> None:
        if workspace.exists():
            shutil.rmtree(workspace)
        settings_path = _ensure_graphrag_workspace(
            workspace, graphrag_llm_model, args.embed_model, graphrag_cli
        )
        _patch_graphrag_settings(
            settings_path,
            graphrag_llm_base_url,
            graphrag_llm_model,
            args.embed_base_url,
            args.embed_model,
            args.embed_encoding_format,
            embed_vector_size,
            args.max_tokens,
            args.temperature,
            relaxed_pruning,
            max_context_tokens,
        )
        input_dir = workspace / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / filename).write_text(content, encoding="utf-8")

    def _index_workspace(workspace: Path, content: str, filename: str) -> None:
        primary_method = str(args.graphrag_index_method)
        index_methods = [primary_method]
        if (
            bool(getattr(args, "graphrag_retry_standard_on_failure", True))
            and primary_method != "standard"
        ):
            index_methods.append("standard")

        last_error: Optional[Exception] = None
        for idx, index_method in enumerate(index_methods):
            if idx > 0:
                _prepare_workspace(workspace, content, filename)
            try:
                _run_graphrag_cmd(
                    [graphrag_cli, "index", "--root", str(workspace), "--method", index_method],
                    check=True,
                )
                return
            except RuntimeError as exc:
                last_error = exc
                if idx + 1 < len(index_methods):
                    reason = _last_non_empty_line(str(exc))
                    print(
                        f"WARNING: GraphRAG {index_method} indexing failed for {workspace.name}; retrying with {index_methods[idx + 1]}. {reason}",
                        file=sys.stderr,
                        flush=True,
                    )
        if last_error is not None:
            raise last_error

    if mode == "merged":
        workspace = out_dir / "graphrag_workspace_merged"
        merged_text = []
        for rec in records:
            merged_text.append(build_context_text(rec.get("context", [])))
        merged_content = "\n\n".join(merged_text)
        _prepare_workspace(workspace, merged_content, "hotpot_merged.txt")
        _index_workspace(workspace, merged_content, "hotpot_merged.txt")

        def _query_record(rec: Dict[str, Any]) -> Tuple[str, str]:
            qid = str(rec.get("_id") or rec.get("id"))
            question = rec.get("question", "")
            result = _run_graphrag_cmd(
                [graphrag_cli, "query", "--root", str(workspace), "--method", "local", "--response-type", "Short Answer", question],
                check=True,
            )
            return qid, _last_non_empty_line(result.stdout or "")

        preds: Dict[str, str] = {}
        if workers > 1 and len(records) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(records))) as executor:
                futures = [executor.submit(_query_record, rec) for rec in records]
                for fut in as_completed(futures):
                    qid, answer = fut.result()
                    preds[qid] = answer
        else:
            for rec in records:
                qid, answer = _query_record(rec)
                preds[qid] = answer
        write_official_pred(preds, pred_path)
        return pred_path

    def _process_record(rec: Dict[str, Any]) -> Tuple[str, str]:
        qid = str(rec.get("_id") or rec.get("id"))
        question = rec.get("question", "")
        context_text = build_context_text(rec.get("context", []))
        workspace = out_dir / f"graphrag_workspace_{qid}"
        try:
            _prepare_workspace(workspace, context_text, "doc.txt")
            _index_workspace(workspace, context_text, "doc.txt")
            result = _run_graphrag_cmd(
                [graphrag_cli, "query", "--root", str(workspace), "--method", "local", "--response-type", "Short Answer", question],
                check=True,
            )
            return qid, _last_non_empty_line(result.stdout or "")
        except Exception as exc:
            if "graphrag cli" in str(exc).lower():
                raise
            if bool(getattr(args, "graphrag_fail_fast", False)):
                raise
            reason = _last_non_empty_line(str(exc))
            print(
                f"WARNING: GraphRAG failed for qid={qid}; returning empty answer. {reason}",
                file=sys.stderr,
                flush=True,
            )
            return qid, ""

    preds: Dict[str, str] = {}
    if workers > 1 and len(records) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(records))) as executor:
            futures = [executor.submit(_process_record, rec) for rec in records]
            for fut in as_completed(futures):
                qid, answer = fut.result()
                preds[qid] = answer
    else:
        for rec in records:
            qid, answer = _process_record(rec)
            preds[qid] = answer
    write_official_pred(preds, pred_path)
    return pred_path


async def run_lightrag(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    out_dir: Path,
    mode: str,
) -> Path:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import setup_logger, wrap_embedding_func_with_attrs

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "lightrag_pred.json"
    workers = max(1, int(getattr(args, "question_workers", 1)))
    setup_logger("lightrag", level="INFO")

    chat_client = VLLMChatClient(
        base_url=args.llm_base_url,
        api_key=args.api_key,
        model=args.llm_model,
        tokenizer_model=args.tokenizer_model,
        max_context_len=args.max_context_len,
        safety_margin_tokens=args.safety_margin_tokens,
    )
    embed_client = OpenAI(base_url=args.embed_base_url, api_key=args.api_key)

    sample_vec = embed_client.embeddings.create(
        input=["probe"],
        model=args.embed_model,
        encoding_format=args.embed_encoding_format,
    ).data[0].embedding
    embedding_dim = len(sample_vec)

    @wrap_embedding_func_with_attrs(
        embedding_dim=embedding_dim,
        max_token_size=args.embed_max_tokens,
        model_name=args.embed_model,
    )
    async def embedding_func(texts: List[str]) -> np.ndarray:
        def _call() -> np.ndarray:
            resp = embed_client.embeddings.create(
                input=texts,
                model=args.embed_model,
                encoding_format=args.embed_encoding_format,
            )
            return np.asarray([item.embedding for item in resp.data], dtype="float32")
        return await asyncio.to_thread(_call)

    async def llm_model_func(prompt: str, **kwargs: Any) -> str:
        explicit_max_tokens = kwargs.get("max_tokens")
        temperature = float(kwargs.get("temperature", args.temperature))
        system_prompt = kwargs.get("system_prompt")
        history_messages = kwargs.get("history_messages") or []
        if not isinstance(history_messages, list):
            history_messages = []

        prompt_text = prompt if isinstance(prompt, str) else str(prompt)
        system_text = str(system_prompt) if system_prompt else ""
        extract_hint = "knowledge graph specialist" in system_text.lower() and "extracting entities and relationships" in system_text.lower()
        if not extract_hint:
            prompt_lower = prompt_text.lower()
            extract_hint = (
                "extract entities and relationships" in prompt_lower
                or "missed or incorrectly formatted" in prompt_lower
            )

        if explicit_max_tokens is not None:
            max_tokens = int(explicit_max_tokens)
        elif extract_hint:
            max_tokens = int(args.lightrag_extract_max_tokens)
        else:
            max_tokens = int(args.max_tokens)

        # Some LightRAG extraction paths pass conservative max_tokens (e.g., 2048).
        # For extraction calls, honor the larger extraction-specific cap from config.
        if extract_hint:
            max_tokens = max(max_tokens, int(args.lightrag_extract_max_tokens))

        prompt = chat_client.truncate_user_prompt(
            user_prompt=prompt_text,
            max_tokens=max_tokens,
            system_prompt=system_text if system_prompt else None,
            history_messages=history_messages,
        )
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        for msg in history_messages:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(role, str) and isinstance(content, str):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        result = await asyncio.to_thread(
            chat_client.chat,
            messages,
            max_tokens,
            temperature,
        )
        if extract_hint:
            return _repair_lightrag_extraction_output(result)
        return result

    async def _build_rag(work_dir: Path) -> Any:
        rag = LightRAG(
            working_dir=str(work_dir),
            llm_model_func=llm_model_func,
            llm_model_name=args.llm_model,
            embedding_func=embedding_func,
            entity_extract_max_gleaning=int(args.lightrag_max_gleaning),
            chunk_token_size=int(args.lightrag_chunk_token_size),
            chunk_overlap_token_size=int(args.lightrag_chunk_overlap_token_size),
        )
        # lightrag_hku>=1.4.x requires explicit storage initialization.
        if hasattr(rag, "initialize_storages"):
            await rag.initialize_storages()
        return rag

    async def _finalize_rag(rag: Any) -> None:
        if hasattr(rag, "finalize_storages"):
            await rag.finalize_storages()

    preds: Dict[str, str] = {}

    if mode == "merged":
        work_dir = out_dir / "lightrag_workspace_merged"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        rag = await _build_rag(work_dir)
        try:
            for rec in records:
                ctx_text = build_context_text(rec.get("context", []))
                await rag.ainsert(ctx_text)
            for rec in records:
                qid = str(rec.get("_id") or rec.get("id"))
                question = rec.get("question", "")
                result = await rag.aquery(
                    question, param=QueryParam(mode="hybrid", enable_rerank=False)
                )
                preds[qid] = str(result).strip()
        finally:
            await _finalize_rag(rag)
        write_official_pred(preds, pred_path)
        return pred_path

    async def _process_record(rec: Dict[str, Any]) -> Tuple[str, str]:
        qid = str(rec.get("_id") or rec.get("id"))
        question = rec.get("question", "")
        ctx_text = build_context_text(rec.get("context", []))
        work_dir = out_dir / f"lightrag_workspace_{qid}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        rag = await _build_rag(work_dir)
        try:
            await rag.ainsert(ctx_text)
            result = await rag.aquery(
                question, param=QueryParam(mode="hybrid", enable_rerank=False)
            )
            return qid, str(result).strip()
        finally:
            await _finalize_rag(rag)

    if workers > 1 and len(records) > 1:
        sem = asyncio.Semaphore(min(workers, len(records)))

        async def _bounded(rec: Dict[str, Any]) -> Tuple[str, str]:
            async with sem:
                return await _process_record(rec)

        results = await asyncio.gather(*[_bounded(rec) for rec in records])
        for qid, answer in results:
            preds[qid] = answer
    else:
        for rec in records:
            qid, answer = await _process_record(rec)
            preds[qid] = answer

    write_official_pred(preds, pred_path)
    return pred_path


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", help="YAML config path")
    pre_args, _ = pre.parse_known_args()
    cfg: Dict[str, Any] = {}
    if pre_args.config:
        cfg_path = Path(pre_args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    def _cfg(key: str, default: Any) -> Any:
        return cfg.get(key, default)

    def _cfg_bool(key: str, default: bool) -> bool:
        value = _cfg(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return bool(value)

    parser = argparse.ArgumentParser(description="HotpotQA comparison runner (RAPTOR/GraphRAG/LightRAG)")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--data", default=_cfg("data", None), help="HotpotQA JSONL path (with context)")
    parser.add_argument("--out_dir", default=_cfg("out_dir", "result/hotpot_compare"), help="Output root")
    parser.add_argument("--method", default=_cfg("method", "all"), choices=["all", "raptor", "graphrag", "lightrag"])
    parser.add_argument("--mode", default=_cfg("mode", "strict"), choices=["strict", "merged"])
    parser.add_argument("--limit", type=int, default=_cfg("limit", 50))
    parser.add_argument("--llm_base_url", default=_cfg("llm_base_url", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--llm_model", default=_cfg("llm_model", "qwen3-30b-a3b"))
    parser.add_argument(
        "--graphrag_llm_base_url",
        default=_cfg("graphrag_llm_base_url", None),
        help="Optional GraphRAG-only LLM endpoint override.",
    )
    parser.add_argument(
        "--graphrag_llm_model",
        default=_cfg("graphrag_llm_model", None),
        help="Optional GraphRAG-only LLM model override.",
    )
    parser.add_argument(
        "--graphrag_cli",
        default=_cfg("graphrag_cli", None),
        help="Path to graphrag executable (optional).",
    )
    parser.add_argument("--embed_base_url", default=_cfg("embed_base_url", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--embed_model", default=_cfg("embed_model", "qwen3-embedding"))
    parser.add_argument("--tokenizer_model", default=_cfg("tokenizer_model", "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"))
    parser.add_argument("--max_context_len", type=int, default=_cfg("max_context_len", 8192))
    parser.add_argument("--safety_margin_tokens", type=int, default=_cfg("safety_margin_tokens", 256))
    parser.add_argument("--max_tokens", type=int, default=_cfg("max_tokens", 256))
    parser.add_argument(
        "--lightrag_extract_max_tokens",
        type=int,
        default=int(_cfg("lightrag_extract_max_tokens", 4096)),
        help="Max tokens for LightRAG entity/relation extraction calls.",
    )
    parser.add_argument(
        "--lightrag_max_gleaning",
        type=int,
        default=int(_cfg("lightrag_max_gleaning", 0)),
        help="Extra LightRAG extraction passes for missed entities/relations (0 disables gleaning).",
    )
    parser.add_argument(
        "--lightrag_chunk_token_size",
        type=int,
        default=int(_cfg("lightrag_chunk_token_size", 600)),
        help="Chunk token size used by LightRAG document splitting.",
    )
    parser.add_argument(
        "--lightrag_chunk_overlap_token_size",
        type=int,
        default=int(_cfg("lightrag_chunk_overlap_token_size", 80)),
        help="Chunk overlap token size used by LightRAG document splitting.",
    )
    parser.add_argument("--temperature", type=float, default=_cfg("temperature", 0.0))
    parser.add_argument("--embed_max_tokens", type=int, default=_cfg("embed_max_tokens", 8192))
    parser.add_argument(
        "--embed_encoding_format",
        default=_cfg("embed_encoding_format", "float"),
        choices=["float", "base64"],
        help="Encoding format for embedding requests (needed for some OpenAI-compatible servers).",
    )
    parser.add_argument("--api_key", default=_cfg("api_key", os.getenv("OPENAI_API_KEY", "EMPTY")))
    parser.add_argument("--raptor_python", default=_cfg("raptor_python", os.getenv("RAPTOR_PYTHON")), help="Python path for RAPTOR (optional)")
    parser.add_argument("--parallel", action="store_true", help="Run raptor/graphrag/lightrag in parallel (method=all)")
    parser.add_argument(
        "--question_workers",
        type=int,
        default=int(_cfg("question_workers", 0)),
        help="Per-method per-question parallel workers in strict mode (0=auto).",
    )
    parser.add_argument(
        "--graphrag_schema_strategy",
        default=str(_cfg("graphrag_schema_strategy", "force")),
        choices=["strict", "force", "fallback"],
        help="GraphRAG structured-output policy when LiteLLM says schema is unsupported: strict=raise, force=bypass guard and keep schema, fallback=force then JSON fallback.",
    )
    parser.add_argument(
        "--graphrag_index_method",
        default=str(_cfg("graphrag_index_method", "standard")),
        choices=["fast", "standard"],
        help="Primary GraphRAG indexing mode.",
    )
    parser.set_defaults(
        graphrag_retry_standard_on_failure=_cfg_bool(
            "graphrag_retry_standard_on_failure", True
        )
    )
    parser.add_argument(
        "--graphrag_retry_standard_on_failure",
        dest="graphrag_retry_standard_on_failure",
        action="store_true",
        help="Retry GraphRAG indexing with standard mode if primary mode fails.",
    )
    parser.add_argument(
        "--no_graphrag_retry_standard_on_failure",
        dest="graphrag_retry_standard_on_failure",
        action="store_false",
        help="Disable GraphRAG retry to standard mode on indexing failure.",
    )
    parser.set_defaults(graphrag_fail_fast=_cfg_bool("graphrag_fail_fast", True))
    parser.add_argument(
        "--graphrag_fail_fast",
        dest="graphrag_fail_fast",
        action="store_true",
        help="Abort run when a single GraphRAG sample fails.",
    )
    parser.add_argument(
        "--no_graphrag_fail_fast",
        dest="graphrag_fail_fast",
        action="store_false",
        help="Keep running and emit empty answer when a single GraphRAG sample fails.",
    )
    parser.set_defaults(
        graphrag_relaxed_pruning=_cfg_bool("graphrag_relaxed_pruning", True)
    )
    parser.add_argument(
        "--graphrag_relaxed_pruning",
        dest="graphrag_relaxed_pruning",
        action="store_true",
        help="Use less aggressive prune_graph thresholds (recommended for single-document strict mode).",
    )
    parser.add_argument(
        "--no_graphrag_relaxed_pruning",
        dest="graphrag_relaxed_pruning",
        action="store_false",
        help="Use default GraphRAG prune_graph thresholds.",
    )
    parser.add_argument(
        "--raptor_workers",
        type=int,
        default=int(_cfg("raptor_workers", 0)),
        help="RAPTOR shard workers in strict mode (0=follow question_workers).",
    )
    parser.add_argument(
        "--raptor_threads",
        type=int,
        default=int(_cfg("raptor_threads", 1)),
        help="Max BLAS/OMP threads per RAPTOR worker process.",
    )
    parser.add_argument(
        "--raptor_per_example_retries",
        type=int,
        default=int(_cfg("raptor_per_example_retries", 1)),
        help="Retries per example inside RAPTOR worker before marking that example failed.",
    )
    parser.set_defaults(
        raptor_fail_on_example_error=_cfg_bool("raptor_fail_on_example_error", False)
    )
    parser.add_argument(
        "--raptor_fail_on_example_error",
        dest="raptor_fail_on_example_error",
        action="store_true",
        help="Fail RAPTOR shard immediately when a single example errors.",
    )
    parser.add_argument(
        "--no_raptor_fail_on_example_error",
        dest="raptor_fail_on_example_error",
        action="store_false",
        help="Continue RAPTOR shard on per-example errors and keep partial outputs.",
    )
    args = parser.parse_args()

    if args.question_workers <= 0:
        if args.parallel:
            args.question_workers = max(1, min(8, os.cpu_count() or 1))
        else:
            args.question_workers = 1
    if args.raptor_workers <= 0:
        args.raptor_workers = args.question_workers
    if args.raptor_threads <= 0:
        args.raptor_threads = 1
    if args.raptor_per_example_retries < 0:
        args.raptor_per_example_retries = 0

    args.repo_root = Path(__file__).resolve().parents[2]
    if not args.data:
        raise ValueError("Dataset path missing. Provide --data or set it in config.")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_hotpot_jsonl(data_path, args.limit)
    gold_path = out_dir / "gold_official.json"
    write_official_gold(records, gold_path)

    run_methods = [args.method] if args.method != "all" else ["raptor", "graphrag", "lightrag"]

    if args.method == "all" and args.parallel:
        base_cmd = [sys.executable, str(Path(__file__).resolve())]
        if args.config:
            base_cmd += ["--config", str(args.config)]
        # Explicitly pass runtime args to avoid config drift.
        base_cmd += [
            "--data",
            str(args.data),
            "--out_dir",
            str(args.out_dir),
            "--mode",
            str(args.mode),
            "--limit",
            str(args.limit),
            "--llm_base_url",
            str(args.llm_base_url),
            "--llm_model",
            str(args.llm_model),
            "--graphrag_schema_strategy",
            str(args.graphrag_schema_strategy),
            "--graphrag_index_method",
            str(args.graphrag_index_method),
            "--embed_base_url",
            str(args.embed_base_url),
            "--embed_model",
            str(args.embed_model),
            "--tokenizer_model",
            str(args.tokenizer_model),
            "--max_context_len",
            str(args.max_context_len),
            "--safety_margin_tokens",
            str(args.safety_margin_tokens),
            "--max_tokens",
            str(args.max_tokens),
            "--lightrag_extract_max_tokens",
            str(args.lightrag_extract_max_tokens),
            "--lightrag_max_gleaning",
            str(args.lightrag_max_gleaning),
            "--lightrag_chunk_token_size",
            str(args.lightrag_chunk_token_size),
            "--lightrag_chunk_overlap_token_size",
            str(args.lightrag_chunk_overlap_token_size),
            "--temperature",
            str(args.temperature),
            "--embed_max_tokens",
            str(args.embed_max_tokens),
            "--embed_encoding_format",
            str(args.embed_encoding_format),
            "--question_workers",
            str(args.question_workers),
            "--raptor_workers",
            str(args.raptor_workers),
            "--raptor_threads",
            str(args.raptor_threads),
            "--raptor_per_example_retries",
            str(args.raptor_per_example_retries),
        ]
        if args.graphrag_retry_standard_on_failure:
            base_cmd += ["--graphrag_retry_standard_on_failure"]
        else:
            base_cmd += ["--no_graphrag_retry_standard_on_failure"]
        if args.graphrag_fail_fast:
            base_cmd += ["--graphrag_fail_fast"]
        else:
            base_cmd += ["--no_graphrag_fail_fast"]
        if args.graphrag_relaxed_pruning:
            base_cmd += ["--graphrag_relaxed_pruning"]
        else:
            base_cmd += ["--no_graphrag_relaxed_pruning"]
        if args.api_key:
            base_cmd += ["--api_key", str(args.api_key)]
        if args.raptor_python:
            base_cmd += ["--raptor_python", str(args.raptor_python)]
        if args.raptor_fail_on_example_error:
            base_cmd += ["--raptor_fail_on_example_error"]
        else:
            base_cmd += ["--no_raptor_fail_on_example_error"]
        if args.graphrag_llm_base_url:
            base_cmd += ["--graphrag_llm_base_url", str(args.graphrag_llm_base_url)]
        if args.graphrag_llm_model:
            base_cmd += ["--graphrag_llm_model", str(args.graphrag_llm_model)]
        if args.graphrag_cli:
            base_cmd += ["--graphrag_cli", str(args.graphrag_cli)]

        method_logs_dir = out_dir / "_parallel_method_logs"
        method_logs_dir.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        proc_infos = []
        for method in run_methods:
            cmd = base_cmd + ["--method", method]
            log_path = method_logs_dir / f"{method}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                env=child_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            proc_infos.append((method, proc, log_path, log_handle))

        for _, proc, _, _ in proc_infos:
            proc.wait()
        for _, _, _, log_handle in proc_infos:
            log_handle.close()
        if any(proc.returncode != 0 for _, proc, _, _ in proc_infos):
            failed_details = []
            for method, proc, log_path, _ in proc_infos:
                if proc.returncode == 0:
                    continue
                failed_details.append(
                    f"[method={method} exit={proc.returncode} log={log_path}]\n"
                    f"{_tail_text(log_path, max_lines=120)}"
                )
            summary = ", ".join(
                [f"{method}:{proc.returncode}" for method, proc, _, _ in proc_infos]
            )
            raise RuntimeError(
                f"Parallel run failed: [{summary}]\n\n" + "\n\n".join(failed_details)
            )
        return

    for method in run_methods:
        method_dir = out_dir / f"{method}_{args.mode}"
        method_dir.mkdir(parents=True, exist_ok=True)
        if method == "raptor":
            pred_path = run_raptor(args, records, method_dir)
        elif method == "graphrag":
            pred_path = run_graphrag(args, records, method_dir, args.mode)
        elif method == "lightrag":
            pred_path = asyncio.run(run_lightrag(args, records, method_dir, args.mode))
        else:
            raise ValueError(f"Unknown method: {method}")
        metrics_path = method_dir / "metrics.json"
        evaluate_official(pred_path, gold_path, metrics_path, args.repo_root)


if __name__ == "__main__":
    main()
