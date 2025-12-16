import os
import sys
import subprocess
import time
import re
import signal
import threading
import torch

def get_env_var(name, default):
    return os.environ.get(name, default)

def log(msg):
    print(f"[Launcher] {msg}", flush=True)

def start_vllm_managed():
    # 1. Detect GPU count
    try:
        n_gpu = torch.cuda.device_count()
    except Exception:
        n_gpu = 1 # Fallback
    
    if n_gpu < 1:
        log("No GPUs detected, forcing tensor_parallel_size=1")
        n_gpu = 1
    
    log(f"Detected {n_gpu} GPUs.")

    # 2. Configuration
    model_name = get_env_var("QWEN3_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    served_name = get_env_var("SERVED_MODEL_NAME", "qwen-served")
    port = get_env_var("VLLM_PORT", "8001")
    
    # Base command
    # Using 'vllm serve' as requested, assuming vllm is in PATH. 
    # If not, we might need sys.executable -m vllm.entrypoints.openai.api_server
    # But user template specifically used 'vllm serve'.
    base_cmd = [
        "vllm", "serve", model_name,
        "--served-model-name", served_name,
        "--host", "0.0.0.0",
        "--port", port,
        "--uvicorn-log-level", "info",
        "--download-dir", "/root/.cache/huggingface",
        "--tensor-parallel-size", str(n_gpu),
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "4096",
        "--limit-mm-per-prompt.image", "1",
        "--limit-mm-per-prompt.video", "0",
        "--swap-space", "16",
        "--no-enforce-eager",
        "--max-num-seqs", "64",
        "--max-num-batched-tokens", "16384",
        "--enable-chunked-prefill",
    ]

    # Critical parameters that should NOT be removed even if unrecognized (heuristic)
    CRITICAL_ARGS = {
        "--tensor-parallel-size", 
        "--max-model-len", 
        "--gpu-memory-utilization",
        "--model",
        "--port",
        "--host"
    }

    current_cmd = list(base_cmd)
    
    while True:
        log(f"Starting vLLM with command: {current_cmd}")
        
        # 3. Subprocess execution (List form, no shell=True)
        process = subprocess.Popen(
            current_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd()
        )
        
        # Monitor startup for errors
        unrecognized_args = []
        startup_success = False
        
        # We need to read stdout in a non-blocking way or thread to avoid deadlocks if we want to kill it early
        # But for simplicity, we read line by line.
        
        try:
            for line in iter(process.stdout.readline, ''):
                print(line, end='') # Passthrough logs
                
                # Check for "unrecognized arguments"
                # vLLM/argparse typically prints: "error: unrecognized arguments: --foo --bar"
                if "unrecognized arguments:" in line:
                    # Extract args
                    # Example: "vllm serve: error: unrecognized arguments: --enable-chunked-prefill"
                    parts = line.split("unrecognized arguments:")
                    if len(parts) > 1:
                        args_str = parts[1].strip()
                        # Split by space, but handle values?
                        # Usually unrecognized args are flags that are not defined.
                        # Simple split might work for flags like --enable-chunked-prefill
                        # But for --foo bar, it might be tricky.
                        # We will try to identify flag-like tokens.
                        candidates = [x for x in args_str.split() if x.startswith("--")]
                        unrecognized_args.extend(candidates)
                
                # Check for success signal (Uvicorn running)
                if "Uvicorn running on" in line:
                    startup_success = True
                    break
                
                # Check if process exited
                if process.poll() is not None:
                    break
        except KeyboardInterrupt:
            process.terminate()
            return

        # If we broke out of loop
        if startup_success:
            log("vLLM started successfully. Streaming logs...")
            # Continue streaming
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
            process.wait()
            return

        # Process exited or we caught error
        return_code = process.poll()
        if return_code is None:
            # It's still running but we haven't seen success yet?
            # Or maybe we found unrecognized args and want to restart?
            if unrecognized_args:
                log(f"Detected unrecognized arguments: {unrecognized_args}")
                process.terminate()
                process.wait()
            else:
                # Keep waiting if no error
                # (The loop above breaks on success or exit, so we only get here if exit or success)
                # If we are here and startup_success is False, it means process exited.
                pass
        
        if unrecognized_args:
            # Remove unsupported args
            new_cmd = []
            skip_next = False
            removed_any = False
            
            # This is a naive removal. It assumes flags are separate items in the list.
            # But arguments with values (e.g. --foo bar) are two items in the list.
            # If 'unrecognized arguments' reports '--foo', we need to remove '--foo' AND 'bar' if it exists?
            # Argparse usually reports the flag.
            
            # We need to iterate over current_cmd and remove matches.
            i = 0
            while i < len(current_cmd):
                arg = current_cmd[i]
                if arg in unrecognized_args:
                    if arg in CRITICAL_ARGS:
                        log(f"Critical argument {arg} is unrecognized! Aborting.")
                        sys.exit(1)
                    
                    log(f"Removing unsupported argument: {arg}")
                    removed_any = True
                    # Check if next item is a value (not starting with -)
                    if i + 1 < len(current_cmd) and not current_cmd[i+1].startswith("-"):
                        log(f"Removing associated value: {current_cmd[i+1]}")
                        i += 2
                    else:
                        i += 1
                else:
                    new_cmd.append(arg)
                    i += 1
            
            if not removed_any:
                log("Could not remove any arguments despite detection. Exiting to avoid infinite loop.")
                sys.exit(1)
                
            current_cmd = new_cmd
            log("Retrying with updated command...")
            time.sleep(1)
            continue
        else:
            log(f"vLLM exited with code {return_code}. No unrecognized arguments detected.")
            sys.exit(return_code)

if __name__ == "__main__":
    start_vllm_managed()
