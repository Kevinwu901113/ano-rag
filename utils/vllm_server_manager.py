import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


@dataclass
class VLLMServerSpec:
    """Definition of a single vLLM server instance to launch."""

    name: str
    port: int
    model: str
    cuda_devices: Optional[str] = None
    host: str = "127.0.0.1"
    quantization: Optional[str] = None
    dtype: Optional[str] = None
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    download_dir: Optional[str] = None
    tensor_parallel_size: Optional[int] = None
    extra_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)


class VLLMServerManager:
    """Context manager that launches and tears down vLLM OpenAI servers.

    The manager reads server specifications from the provided configuration
    dictionary. Each server is spawned as a subprocess and monitored until the
    OpenAI-compatible `/v1/models` endpoint becomes available.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        autostart_cfg = config_dict or {}
        self.enabled: bool = bool(autostart_cfg.get("enabled", False))
        self.readiness_timeout: int = int(autostart_cfg.get("readiness_timeout", 180))
        self.health_interval: float = float(autostart_cfg.get("health_interval", 2.0))
        self.log_dir: Path = Path(autostart_cfg.get("log_dir", "logs/vllm"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        servers_cfg = autostart_cfg.get("servers", [])
        self.servers: List[VLLMServerSpec] = []
        for idx, raw in enumerate(servers_cfg):
            if not isinstance(raw, dict):
                logger.warning(f"Skipping invalid vLLM server config at index {idx}: expected dict, got {type(raw).__name__}")
                continue
            name = raw.get("name") or f"vllm_{idx}"
            port = raw.get("port")
            model = raw.get("model")
            if port is None or model is None:
                logger.warning(f"Skipping vLLM server {name}: 'port' and 'model' are required")
                continue
            spec = VLLMServerSpec(
                name=name,
                port=int(port),
                model=str(model),
                cuda_devices=raw.get("cuda_devices"),
                host=raw.get("host", "127.0.0.1"),
                quantization=raw.get("quantization"),
                dtype=raw.get("dtype"),
                gpu_memory_utilization=raw.get("gpu_memory_utilization"),
                max_model_len=raw.get("max_model_len"),
                max_num_batched_tokens=raw.get("max_num_batched_tokens"),
                download_dir=raw.get("download_dir"),
                tensor_parallel_size=raw.get("tensor_parallel_size"),
                extra_args=list(raw.get("extra_args", [])),
                env=dict(raw.get("env", {})),
            )
            self.servers.append(spec)

        self._processes: List[Dict[str, Any]] = []
        self._ready_processes: Dict[int, Dict[str, Any]] = {}
        self._failed_processes: List[tuple[Dict[str, Any], str]] = []
        self._ready_endpoints: List[str] = []

    # -- context-manager protocol ------------------------------------------------
    def __enter__(self):
        if self.enabled:
            self.start_all()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_all()
        return False

    # -- public API --------------------------------------------------------------
    def start_all(self) -> None:
        """Launch all configured vLLM servers and wait for readiness."""
        if not self.servers:
            logger.warning("vLLM autostart enabled but no servers configured; skipping launch")
            return

        # Reset readiness bookkeeping for this launch cycle
        self._ready_processes = {}
        self._failed_processes = []
        self._ready_endpoints = []

        python_executable = sys.executable or "python"
        for spec in self.servers:
            cmd = self._build_command(python_executable, spec)
            env = self._build_env(spec)
            log_path = self.log_dir / f"{spec.name}.log"
            logger.info(
                "Starting vLLM server '{}' on {}:{} (model={}) -> log: {}",
                spec.name,
                spec.host,
                spec.port,
                spec.model,
                log_path,
            )
            log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=self._infer_repo_root(),
            )
            self._processes.append(
                {"process": process, "log_file": log_file, "spec": spec, "log_path": log_path}
            )

        ready_map, failures = self._wait_until_ready()
        self._ready_processes = ready_map
        self._failed_processes = failures
        self._ready_endpoints = [
            f"http://{info['spec'].host}:{info['spec'].port}/v1"
            for info in ready_map.values()
        ]

    def stop_all(self) -> None:
        """Terminate all spawned vLLM servers."""
        for proc_info in reversed(self._processes):
            process = proc_info["process"]
            log_file = proc_info["log_file"]
            spec: VLLMServerSpec = proc_info["spec"]

            if process.poll() is not None:
                logger.info("vLLM server '%s' already exited (code %s)", spec.name, process.returncode)
                log_file.close()
                continue

            logger.info("Stopping vLLM server '%s' (pid=%s)", spec.name, process.pid)
            try:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    logger.warning("vLLM server '%s' did not terminate gracefully; killing", spec.name)
                    process.kill()
                    process.wait(timeout=10)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to stop vLLM server '%s': %s", spec.name, exc)
            finally:
                log_file.close()

        self._processes.clear()
        self._ready_processes = {}
        self._failed_processes = []
        self._ready_endpoints = []

    def get_ready_endpoints(self) -> List[str]:
        """Return a list of OpenAI-compatible endpoints for ready servers."""
        return list(self._ready_endpoints)

    def get_failed_servers(self) -> List[str]:
        """Return human-readable descriptions of servers that failed to start."""
        failures: List[str] = []
        for proc_info, reason in self._failed_processes:
            spec: VLLMServerSpec = proc_info["spec"]
            failures.append(f"{spec.name}@{spec.host}:{spec.port} ({reason})")
        return failures

    # -- internal helpers --------------------------------------------------------
    def _build_command(self, python_executable: str, spec: VLLMServerSpec) -> List[str]:
        cmd: List[str] = [
            python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            spec.model,
            "--port",
            str(spec.port),
            "--host",
            spec.host,
        ]
        if spec.quantization:
            cmd.extend(["--quantization", str(spec.quantization)])
        if spec.dtype:
            cmd.extend(["--dtype", str(spec.dtype)])
        if spec.gpu_memory_utilization is not None:
            cmd.extend(
                ["--gpu-memory-utilization", str(spec.gpu_memory_utilization)]
            )
        if spec.max_model_len is not None:
            cmd.extend(["--max-model-len", str(spec.max_model_len)])
        if spec.max_num_batched_tokens is not None:
            cmd.extend(
                ["--max-num-batched-tokens", str(spec.max_num_batched_tokens)]
            )
        if spec.download_dir:
            cmd.extend(["--download-dir", str(spec.download_dir)])
        if spec.tensor_parallel_size is not None:
            cmd.extend(
                ["--tensor-parallel-size", str(spec.tensor_parallel_size)]
            )
        if spec.extra_args:
            cmd.extend([str(arg) for arg in spec.extra_args])
        return cmd

    def _build_env(self, spec: VLLMServerSpec) -> Dict[str, str]:
        env = os.environ.copy()
        if spec.cuda_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(spec.cuda_devices)
        env.update(spec.env or {})
        # Ensure the processes inherit an unbuffered stdout for timely logs
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _wait_until_ready(self) -> tuple[Dict[int, Dict[str, Any]], List[tuple[Dict[str, Any], str]]]:
        """Poll `/v1/models` until at least one server is ready."""
        deadline = time.time() + self.readiness_timeout
        remaining = {proc_info["spec"].port: proc_info for proc_info in self._processes}
        ready_ports: Dict[int, Dict[str, Any]] = {}
        failures: List[tuple[Dict[str, Any], str]] = []

        while remaining:
            now = time.time()
            if now > deadline:
                for info in remaining.values():
                    failures.append((info, "startup timeout"))
                remaining.clear()
                break

            for port in list(remaining.keys()):
                proc_info = remaining[port]
                spec: VLLMServerSpec = proc_info["spec"]
                process = proc_info["process"]
                if not self._is_process_running(process):
                    exit_code = process.returncode
                    reason = f"exited (code {exit_code})" if exit_code is not None else "exited"
                    failures.append((proc_info, reason))
                    remaining.pop(port, None)
                    continue
                if self._probe_server(spec.host, port):
                    logger.info("vLLM server '%s' on %s:%s is ready", spec.name, spec.host, port)
                    ready_ports[port] = proc_info
                    remaining.pop(port, None)
            if remaining:
                time.sleep(self.health_interval)

        if not ready_ports:
            details = "; ".join(
                f"{info['spec'].name}@{info['spec'].host}:{info['spec'].port} ({reason})"
                for info, reason in failures
            ) or "no servers became ready"
            raise RuntimeError(f"Failed to start any vLLM server: {details}")

        if failures:
            for proc_info, reason in failures:
                spec: VLLMServerSpec = proc_info["spec"]
                logger.warning(
                    "vLLM server '%s' on %s:%s unavailable (%s); continuing with remaining instances",
                    spec.name,
                    spec.host,
                    spec.port,
                    reason,
                )

        return ready_ports, failures

    @staticmethod
    def _probe_server(host: str, port: int) -> bool:
        url = f"http://{host}:{port}/v1/models"
        try:
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    @staticmethod
    def _is_process_running(process: subprocess.Popen) -> bool:
        return process.poll() is None

    @staticmethod
    def _infer_repo_root() -> Optional[str]:
        # Attempt to use the project root (parent of utils/)
        current_file = Path(__file__).resolve()
        repo_root = current_file.parent.parent
        return str(repo_root)


class NoOpVLLMServerManager(VLLMServerManager):
    """Fallback manager used when autostart is disabled."""

    def __init__(self):
        super().__init__({})
        self.enabled = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def start_all(self):
        return

    def stop_all(self):
        return

    def get_ready_endpoints(self) -> List[str]:
        return []

    def get_failed_servers(self) -> List[str]:
        return []
