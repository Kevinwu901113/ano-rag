import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_LLM_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_CACHE_DIR = "/home/wjk/.cache/hf"
DEFAULT_PROXY = "http://192.168.192.246:7890"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_LLM_GPU_MEM = 0.70
DEFAULT_LLM_SWAP_SPACE = 8
DEFAULT_LLM_KV_CACHE_DTYPE = "fp8"


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


def _model_cached(model_id: str, cache_dir: Path) -> bool:
    if not model_id:
        return False
    model_path = Path(model_id).expanduser()
    if model_path.exists():
        return True
    repo_key = model_id.replace("/", "--")
    candidates = [
        cache_dir / f"models--{repo_key}",
        cache_dir / "hub" / f"models--{repo_key}",
    ]
    for base in candidates:
        snapshots = base / "snapshots"
        if snapshots.exists() and any(snapshots.iterdir()):
            return True
    return False


def _build_env(
    cache_dir: Path,
    use_proxy: bool,
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
    env["HF_ENDPOINT"] = hf_endpoint
    env["HF_HUB_ENDPOINT"] = hf_endpoint
    if pytorch_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = pytorch_alloc_conf
    if use_proxy:
        env["http_proxy"] = http_proxy
        env["https_proxy"] = https_proxy
    else:
        env.pop("http_proxy", None)
        env.pop("https_proxy", None)
    return env


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
        use_proxy=not cached,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        hf_endpoint=hf_endpoint,
        pytorch_alloc_conf=pytorch_alloc_conf or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    )
    if cached:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
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
    log_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(cmd_str)
    print(f"[audit] wrote {log_path}")

    if dry_run:
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"])

    return subprocess.Popen(cmd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM on two GPUs (LLM + Embedding)")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
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
    args = parser.parse_args()

    if args.pytorch_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.pytorch_alloc_conf

    if not args.dry_run:
        _check_vllm_version((0, 9, 0))
        count = _gpu_count()
        if count < 2:
            raise RuntimeError(f"Need at least 2 visible GPUs, found {count}")
        if args.llm_gpu == args.embed_gpu:
            raise RuntimeError("LLM and embedding GPU must be different")
        if args.llm_gpu >= count or args.embed_gpu >= count:
            raise RuntimeError(f"GPU index out of range (visible GPUs: {count})")

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).expanduser()

    llm_proc = _spawn_server(
        name="qwen3-30b-a3b",
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
        quantization=args.llm_quantization or None,
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
    embed_proc = _spawn_server(
        name="qwen3-embedding",
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

    def _shutdown(*_args) -> None:
        for proc in (llm_proc, embed_proc):
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1.0)
        for proc in (llm_proc, embed_proc):
            if proc.poll() is None:
                proc.kill()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            time.sleep(2.0)
            if llm_proc.poll() is not None or embed_proc.poll() is not None:
                raise RuntimeError("One of the vLLM processes exited")
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
