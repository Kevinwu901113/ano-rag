#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
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
        if hasattr(self.tokenizer, "apply_chat_template"):
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


def run_raptor(args: argparse.Namespace, records: List[Dict[str, Any]], out_dir: Path) -> Path:
    pred_path = out_dir / "raptor_pred.json"
    gold_path = out_dir / "raptor_gold.json"
    cmd = [
        args.raptor_python or sys.executable,
        str(args.repo_root / "RAPTOR" / "raptor" / "run_hotpotqa_raptor_vllm.py"),
        "--data_jsonl",
        str(args.data),
        "--limit",
        str(args.limit),
        "--out_pred",
        str(pred_path),
        "--out_gold",
        str(gold_path),
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
    ]
    subprocess.run(cmd, check=True)
    return pred_path


def _ensure_graphrag_workspace(root: Path, llm_model: str, embed_model: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    settings_path = root / "settings.yaml"
    env_path = root / ".env"
    if shutil.which("graphrag") is None:
        raise RuntimeError("graphrag CLI not found. Install with `pip install graphrag`.")
    if not settings_path.exists():
        subprocess.run(
            ["graphrag", "init", "--root", str(root), "--model", llm_model, "--embedding", embed_model],
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
    max_tokens: int,
    temperature: float,
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

    cfg.setdefault("encoding_model", "cl100k_base")
    settings_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_graphrag(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    out_dir: Path,
    mode: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "graphrag_pred.json"
    graphrag_env = os.environ.copy()
    graphrag_env.setdefault("GRAPHRAG_API_KEY", args.api_key or "EMPTY")

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

    if mode == "merged":
        workspace = out_dir / "graphrag_workspace_merged"
        settings_path = _ensure_graphrag_workspace(workspace, args.llm_model, args.embed_model)
        _patch_graphrag_settings(
            settings_path,
            args.llm_base_url,
            args.llm_model,
            args.embed_base_url,
            args.embed_model,
            args.embed_encoding_format,
            args.max_tokens,
            args.temperature,
        )
        input_dir = workspace / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        merged_text = []
        for rec in records:
            merged_text.append(build_context_text(rec.get("context", [])))
        (input_dir / "hotpot_merged.txt").write_text("\n\n".join(merged_text), encoding="utf-8")
        _run_graphrag_cmd(["graphrag", "index", "--root", str(workspace), "--method", "fast"], check=True)
        preds: Dict[str, str] = {}
        for rec in records:
            qid = str(rec.get("_id") or rec.get("id"))
            question = rec.get("question", "")
            result = _run_graphrag_cmd(
                ["graphrag", "query", "--root", str(workspace), "--method", "local", "--response-type", "Short Answer", question],
                check=False,
            )
            preds[qid] = _last_non_empty_line(result.stdout or "")
        write_official_pred(preds, pred_path)
        return pred_path

    preds: Dict[str, str] = {}
    for rec in records:
        qid = str(rec.get("_id") or rec.get("id"))
        question = rec.get("question", "")
        context_text = build_context_text(rec.get("context", []))
        workspace = out_dir / f"graphrag_workspace_{qid}"
        if workspace.exists():
            shutil.rmtree(workspace)
        settings_path = _ensure_graphrag_workspace(workspace, args.llm_model, args.embed_model)
        _patch_graphrag_settings(
            settings_path,
            args.llm_base_url,
            args.llm_model,
            args.embed_base_url,
            args.embed_model,
            args.embed_encoding_format,
            args.max_tokens,
            args.temperature,
        )
        input_dir = workspace / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "doc.txt").write_text(context_text, encoding="utf-8")
        _run_graphrag_cmd(["graphrag", "index", "--root", str(workspace), "--method", "fast"], check=True)
        result = _run_graphrag_cmd(
            ["graphrag", "query", "--root", str(workspace), "--method", "local", "--response-type", "Short Answer", question],
            check=False,
        )
        preds[qid] = _last_non_empty_line(result.stdout or "")
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
        max_tokens = int(kwargs.get("max_tokens", args.max_tokens))
        temperature = float(kwargs.get("temperature", args.temperature))
        prompt = chat_client.truncate_prompt(prompt, max_tokens)
        messages = [{"role": "user", "content": prompt}]
        return await asyncio.to_thread(chat_client.chat, messages, max_tokens, temperature)

    async def _build_rag(work_dir: Path) -> Any:
        rag = LightRAG(
            working_dir=str(work_dir),
            llm_model_func=llm_model_func,
            llm_model_name=args.llm_model,
            embedding_func=embedding_func,
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
                result = await rag.aquery(question, param=QueryParam(mode="hybrid"))
                preds[qid] = str(result).strip()
        finally:
            await _finalize_rag(rag)
        write_official_pred(preds, pred_path)
        return pred_path

    for rec in records:
        qid = str(rec.get("_id") or rec.get("id"))
        question = rec.get("question", "")
        ctx_text = build_context_text(rec.get("context", []))
        work_dir = out_dir / f"lightrag_workspace_{qid}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        rag = await _build_rag(work_dir)
        try:
            await rag.ainsert(ctx_text)
            result = await rag.aquery(question, param=QueryParam(mode="hybrid"))
            preds[qid] = str(result).strip()
        finally:
            await _finalize_rag(rag)

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

    parser = argparse.ArgumentParser(description="HotpotQA comparison runner (RAPTOR/GraphRAG/LightRAG)")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--data", default=_cfg("data", None), help="HotpotQA JSONL path (with context)")
    parser.add_argument("--out_dir", default=_cfg("out_dir", "result/hotpot_compare"), help="Output root")
    parser.add_argument("--method", default=_cfg("method", "all"), choices=["all", "raptor", "graphrag", "lightrag"])
    parser.add_argument("--mode", default=_cfg("mode", "strict"), choices=["strict", "merged"])
    parser.add_argument("--limit", type=int, default=_cfg("limit", 50))
    parser.add_argument("--llm_base_url", default=_cfg("llm_base_url", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--llm_model", default=_cfg("llm_model", "qwen3-30b-a3b"))
    parser.add_argument("--embed_base_url", default=_cfg("embed_base_url", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--embed_model", default=_cfg("embed_model", "qwen3-embedding"))
    parser.add_argument("--tokenizer_model", default=_cfg("tokenizer_model", "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"))
    parser.add_argument("--max_context_len", type=int, default=_cfg("max_context_len", 8192))
    parser.add_argument("--safety_margin_tokens", type=int, default=_cfg("safety_margin_tokens", 256))
    parser.add_argument("--max_tokens", type=int, default=_cfg("max_tokens", 256))
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
    args = parser.parse_args()

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
            "--temperature",
            str(args.temperature),
            "--embed_max_tokens",
            str(args.embed_max_tokens),
            "--embed_encoding_format",
            str(args.embed_encoding_format),
        ]
        if args.api_key:
            base_cmd += ["--api_key", str(args.api_key)]
        if args.raptor_python:
            base_cmd += ["--raptor_python", str(args.raptor_python)]

        procs = []
        for method in run_methods:
            cmd = base_cmd + ["--method", method]
            procs.append(subprocess.Popen(cmd))
        exit_codes = [proc.wait() for proc in procs]
        if any(code != 0 for code in exit_codes):
            raise RuntimeError(f"Parallel run failed: {exit_codes}")
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
