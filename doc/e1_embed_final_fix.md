# Embedding and Final Tag Fix for E1

## A. Configuration Source Analysis

We analyzed the configuration files to locate the source of the `embedding` configuration.

- **`experiments/configs/e1_main_qwen.yaml`**: This is the main experiment config. It includes `common.yaml`.
- **`experiments/configs/common.yaml`**: This file defines the default `embedding` section:
  ```yaml
  embedding:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cpu"
    batch_size: 4
    max_length: 512
    normalize: true
    dtype: null
  ```

Since `e1_main_qwen.yaml` does not override the `embedding` section, the values from `common.yaml` are used.

## B. Plan for Changes

1.  **Embedding Update**: We will update `experiments/configs/common.yaml` to use `/home/wjk/models/qwen3-emb` as the default embedding model.
2.  **Prompt Update**: We will locate the system prompt used for generation (likely in `structrag/prompts.py` or similar) and enforce the "FINAL:" protocol and disable thinking tags.
3.  **Verification**: We will run a smoke test `e1_smoke_v2` to verify the resolved configuration and the output format.
