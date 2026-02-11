import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_LLM_MODEL = "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_CACHE_DIR = "/home/wjk/.cache/hf"
DEFAULT_PROXY = "http://192.168.192.246:7890"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_LLM_GPU_MEM = 0.70
DEFAULT_LLM_SWAP_SPACE = 8
DEFAULT_LLM_KV_CACHE_DTYPE = "fp8"
WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def _parse_version(raw: str) -> Tuple[int, int, int]:
    if not raw:
        return (0, 0, 0)
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", raw)
    if not match:
        return (0, 0, 0)
    return tuple(int(group) for group in match.groups())


def _check_vllm_version(min_version: Tuple[int, int, int]) -> None:
    try:
        import vllm  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"vLLM not available: {exc}") from exc
    current = _parse_version(getattr(vllm, "__version__", "0.0.0"))
    if current < min_version:
        raise RuntimeError(f"vLLM {current} is below required {min_version}")


def _gpu_count() -> int:
    try:
        import torch  # type: ignore
    except Exception:
        return 0
    return int(torch.cuda.device_count())


def _resolve_snapshots(model_id: str, cache_dir: Path) -> List[Path]:
    model_path = Path(model_id).expanduser()
    if model_path.exists():
        if model_path.is_dir():
            return [model_path]
        return [model_path.parent]
    repo_key = model_id.replace("/", "--")
    candidates = [
        cache_dir / f"models--{repo_key}",
        cache_dir / "hub" / f"models--{repo_key}",
    ]
    snapshots: List[Path] = []
    for base in candidates:
        snap_root = base / "snapshots"
        if snap_root.exists():
            for snapshot in snap_root.iterdir():
                if snapshot.is_dir():
                    snapshots.append(snapshot)
    return snapshots


def _has_weight_files(snapshot_dir: Path) -> bool:
    for suffix in WEIGHT_FILE_SUFFIXES:
        if any(snapshot_dir.rglob(f"*{suffix}")):
            return True
    return False


def _model_cached(model_id: str, cache_dir: Path) -> bool:
    if not model_id:
        return False
    model_path = Path(model_id).expanduser()
    if model_path.exists() and model_path.is_file():
        return model_path.suffix in WEIGHT_FILE_SUFFIXES
    for snapshot in _resolve_snapshots(model_id, cache_dir):
        if _has_weight_files(snapshot):
            return True
    return False


def _detect_quantization_from_config(model_id: str, cache_dir: Path) -> Optional[str]:
    for snapshot in _resolve_snapshots(model_id, cache_dir):
        config_path = snapshot / "config.json"
        if not config_path.exists():
            continue
        try:
            raw = config_path.read_text(encoding="utf-8")
        except Exception:
            continue
        lowered = raw.lower()
        if "compressed-tensors" in lowered or "compressed_tensors" in lowered:
            return "compressed-tensors"
        if "awq" in lowered:
            return "awq"
        if "gptq" in lowered:
            return "gptq"
    return None


def _build_env(
    cache_dir: Path,
    http_proxy: str,
    https_proxy: str,
    hf_endpoint: str,
    pytorch_alloc_conf: Optional[str],
) -> Dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(cache_dir)
    env["HF_HUB_CACHE"] = str(cache_dir)
    env["TRANSFORMERS_CACHE"] = str(cache_dir)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
        env["HF_HUB_ENDPOINT"] = hf_endpoint
    if pytorch_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = pytorch_alloc_conf
    if http_proxy:
        env["http_proxy"] = http_proxy
    else:
        env.pop("http_proxy", None)
    if https_proxy:
        env["https_proxy"] = https_proxy
    else:
        env.pop("https_proxy", None)
    return env


@contextmanager
def _temp_env(overrides: Dict[str, Optional[str]]):
    prior = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _preload_model(
    model_id: str,
    cache_dir: Path,
    http_proxy: str,
    https_proxy: str,
    hf_endpoint: str,
) -> None:
    if not model_id:
        return
    model_path = Path(model_id).expanduser()
    if model_path.exists():
        return
    if _model_cached(model_id, cache_dir):
        print(f"[preload] cache hit for {model_id}")
        return
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub not available: {exc}") from exc

    env_overrides: Dict[str, Optional[str]] = dict(
        _build_env(
            cache_dir,
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            hf_endpoint=hf_endpoint,
            pytorch_alloc_conf=None,
        )
    )
    env_overrides["HF_HUB_OFFLINE"] = None
    env_overrides["TRANSFORMERS_OFFLINE"] = None

    print(f"[preload] downloading {model_id} to {cache_dir}")
    with _temp_env(env_overrides):
        snapshot_download(repo_id=model_id, cache_dir=str(cache_dir), resume_download=True)
    print(f"[preload] completed {model_id}")


def _resolve_quantization(
    model_id: str, requested: Optional[str], cache_dir: Path
) -> Tuple[Optional[str], str]:
    if requested:
        return requested, "explicit"
    if "awq" not in model_id.lower():
        return None, "none"
    detected = _detect_quantization_from_config(model_id, cache_dir)
    if detected:
        if detected == "awq":
            return "awq", "config:awq"
        return None, f"config:{detected}"
    return None, "config-missing"


def _print_env_exports(env: Dict[str, str], label: str) -> None:
    print(f"[env:{label}]")
    keys = [
        "HF_ENDPOINT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "http_proxy",
        "https_proxy",
    ]
    for key in keys:
        value = env.get(key, "")
        print(f"export {key}={value}")


def _spawn_server(
    name: str,
    model_id: str,
    host: str,
    port: int,
    gpu_id: int,
    cache_dir: Path,
    http_proxy: str,
    https_proxy: str,
    hf_endpoint: str,
    gpu_mem_util: float,
    trust_remote_code: bool,
    task: Optional[str],
    quantization: Optional[str],
    quantization_reason: Optional[str],
    log_dir: Path,
    label: str,
    max_model_len: Optional[int],
    max_num_seqs: Optional[int],
    max_num_batched_tokens: Optional[int],
    swap_space: Optional[int],
    cpu_offload_gb: Optional[int],
    kv_cache_dtype: Optional[str],
    pytorch_alloc_conf: Optional[str],
    dry_run: bool,
) -> subprocess.Popen:
    cached = _model_cached(model_id, cache_dir)
    env = _build_env(
        cache_dir,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        hf_endpoint=hf_endpoint,
        pytorch_alloc_conf=pytorch_alloc_conf or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    )
    if cached:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model_id,
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--served-model-name",
        name,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if task:
        cmd.extend(["--task", task])
    if quantization:
        cmd.extend(["--quantization", quantization])
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    if max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
    if swap_space is not None:
        cmd.extend(["--swap-space", str(swap_space)])
    if cpu_offload_gb is not None:
        cmd.extend(["--cpu-offload-gb", str(cpu_offload_gb)])
    if kv_cache_dtype:
        cmd.extend(["--kv-cache-dtype", kv_cache_dtype])

    cmd_str = shlex.join(cmd)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vllm_cmd_gpu{gpu_id}.txt"
    audit_lines = [
        f"label={label}",
        f"model_id={model_id}",
        f"cached={cached}",
        f"quantization_reason={quantization_reason}",
        f"cmd={cmd_str}",
        "env={}",
        f"  CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}",
        f"  HF_HOME={env.get('HF_HOME')}",
        f"  HF_HUB_CACHE={env.get('HF_HUB_CACHE')}",
        f"  TRANSFORMERS_CACHE={env.get('TRANSFORMERS_CACHE')}",
        f"  HF_ENDPOINT={env.get('HF_ENDPOINT')}",
        f"  HF_HUB_ENDPOINT={env.get('HF_HUB_ENDPOINT')}",
        f"  HF_HUB_DISABLE_TELEMETRY={env.get('HF_HUB_DISABLE_TELEMETRY')}",
        f"  HF_HUB_OFFLINE={env.get('HF_HUB_OFFLINE')}",
        f"  TRANSFORMERS_OFFLINE={env.get('TRANSFORMERS_OFFLINE')}",
        f"  http_proxy={env.get('http_proxy')}",
        f"  https_proxy={env.get('https_proxy')}",
        f"  PYTORCH_CUDA_ALLOC_CONF={env.get('PYTORCH_CUDA_ALLOC_CONF')}",
    ]
    _print_env_exports(env, label)
    log_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(cmd_str)
    print(f"[audit] wrote {log_path}")

    if dry_run:
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"])

    return subprocess.Popen(cmd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM on two GPUs (LLM + Embedding)")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-served-model-name", default="qwen3-30b-a3b")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--embed-served-model-name", default="qwen3-embedding")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--llm-port", type=int, default=8000)
    parser.add_argument("--embed-port", type=int, default=8001)
    parser.add_argument("--llm-gpu", type=int, default=0)
    parser.add_argument("--embed-gpu", type=int, default=1)
    parser.add_argument("--llm-gpu-mem", type=float, default=DEFAULT_LLM_GPU_MEM)
    parser.add_argument("--embed-gpu-mem", type=float, default=0.70)
    parser.add_argument("--llm-max-model-len", type=int, default=None)
    parser.add_argument("--embed-max-model-len", type=int, default=None)
    parser.add_argument("--llm-max-num-seqs", type=int, default=None)
    parser.add_argument("--embed-max-num-seqs", type=int, default=None)
    parser.add_argument("--llm-max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--embed-max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--llm-swap-space", type=int, default=DEFAULT_LLM_SWAP_SPACE)
    parser.add_argument("--llm-cpu-offload-gb", type=int, default=None)
    parser.add_argument("--llm-kv-cache-dtype", default=DEFAULT_LLM_KV_CACHE_DTYPE)
    parser.add_argument("--embed-kv-cache-dtype", default="")
    parser.add_argument("--pytorch-alloc-conf", default="")
    parser.add_argument("--http-proxy", default=DEFAULT_PROXY)
    parser.add_argument("--https-proxy", default=DEFAULT_PROXY)
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--embed-task", default="embedding")
    parser.add_argument("--llm-quantization", default="")
    parser.add_argument("--embed-quantization", default="")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm-only", action="store_true", help="Only start the LLM server")
    parser.add_argument("--embed-only", action="store_true", help="Only start the embedding server")
    args = parser.parse_args()

    if not args.http_proxy:
        args.http_proxy = DEFAULT_PROXY
    if not args.https_proxy:
        args.https_proxy = DEFAULT_PROXY
    if not args.hf_endpoint:
        args.hf_endpoint = DEFAULT_HF_ENDPOINT

    if args.pytorch_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.pytorch_alloc_conf

    if args.llm_only and args.embed_only:
        raise RuntimeError("Choose only one of --llm-only or --embed-only")
    start_llm = not args.embed_only
    start_embed = not args.llm_only

    if not args.dry_run:
        _check_vllm_version((0, 9, 0))
        count = _gpu_count()
        if start_llm and start_embed:
            if count < 2:
                raise RuntimeError(f"Need at least 2 visible GPUs, found {count}")
            if args.llm_gpu == args.embed_gpu:
                raise RuntimeError("LLM and embedding GPU must be different")
            if args.llm_gpu >= count or args.embed_gpu >= count:
                raise RuntimeError(f"GPU index out of range (visible GPUs: {count})")
        else:
            if count < 1:
                raise RuntimeError(f"Need at least 1 visible GPU, found {count}")
            gpu_idx = args.llm_gpu if start_llm else args.embed_gpu
            if gpu_idx >= count:
                raise RuntimeError(f"GPU index out of range (visible GPUs: {count})")

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).expanduser()

    if not args.dry_run:
        if start_llm:
            _preload_model(
                args.llm_model,
                cache_dir=cache_dir,
                http_proxy=args.http_proxy,
                https_proxy=args.https_proxy,
                hf_endpoint=args.hf_endpoint,
            )
        if start_embed:
            _preload_model(
                args.embed_model,
                cache_dir=cache_dir,
                http_proxy=args.http_proxy,
                https_proxy=args.https_proxy,
                hf_endpoint=args.hf_endpoint,
            )

    llm_proc = None
    embed_proc = None
    if start_llm:
        llm_quantization, llm_quant_reason = _resolve_quantization(
            args.llm_model, args.llm_quantization or None, cache_dir
        )
        llm_proc = _spawn_server(
            name=args.llm_served_model_name,
            model_id=args.llm_model,
            host=args.host,
            port=args.llm_port,
            gpu_id=args.llm_gpu,
            cache_dir=cache_dir,
            http_proxy=args.http_proxy,
            https_proxy=args.https_proxy,
            hf_endpoint=args.hf_endpoint,
            gpu_mem_util=args.llm_gpu_mem,
            trust_remote_code=args.trust_remote_code,
            task=None,
            quantization=llm_quantization,
            quantization_reason=llm_quant_reason,
            log_dir=log_dir,
            label="llm",
            max_model_len=args.llm_max_model_len,
            max_num_seqs=args.llm_max_num_seqs,
            max_num_batched_tokens=args.llm_max_num_batched_tokens,
            swap_space=args.llm_swap_space,
            cpu_offload_gb=args.llm_cpu_offload_gb,
            kv_cache_dtype=args.llm_kv_cache_dtype or None,
            pytorch_alloc_conf=args.pytorch_alloc_conf or None,
            dry_run=args.dry_run,
        )
    if start_embed:
        embed_proc = _spawn_server(
            name=args.embed_served_model_name,
            model_id=args.embed_model,
            host=args.host,
            port=args.embed_port,
            gpu_id=args.embed_gpu,
            cache_dir=cache_dir,
            http_proxy=args.http_proxy,
            https_proxy=args.https_proxy,
            hf_endpoint=args.hf_endpoint,
            gpu_mem_util=args.embed_gpu_mem,
            trust_remote_code=args.trust_remote_code,
            task=args.embed_task,
            quantization=args.embed_quantization or None,
            quantization_reason="explicit" if args.embed_quantization else "none",
            log_dir=log_dir,
            label="embedding",
            max_model_len=args.embed_max_model_len,
            max_num_seqs=args.embed_max_num_seqs,
            max_num_batched_tokens=args.embed_max_num_batched_tokens,
            swap_space=None,
            cpu_offload_gb=None,
            kv_cache_dtype=args.embed_kv_cache_dtype or None,
            pytorch_alloc_conf=args.pytorch_alloc_conf or None,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        return

    procs = [proc for proc in (llm_proc, embed_proc) if proc is not None]

    def _shutdown(*_args) -> None:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1.0)
        for proc in procs:
            if proc.poll() is None:
                proc.kill()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            time.sleep(2.0)
            if any(proc.poll() is not None for proc in procs):
                raise RuntimeError("One of the vLLM processes exited")
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
