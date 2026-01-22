# NarrativeQA Experiment Runner

## One-command full grid

```bash
python scripts/narrativeqa/run_all_experiments.py \
  --split dev \
  --limit 0
```

Outputs:

```
result/narrativeqa/<run_id>/
  summary.json
  summary.md
  runs/
    relrag_bm25_vllm/
    relrag_bm25_openai/
    ...
```

## Resume

```bash
python scripts/narrativeqa/run_all_experiments.py --run_id <existing> --resume
```

## Run subset

```bash
python scripts/narrativeqa/run_all_experiments.py \
  --run_id <run_id> \
  --only relrag_bm25_vllm,pure_dense_openai
```

## Story-as-context

```bash
python scripts/narrativeqa/run_all_experiments.py \
  --context_mode story \
  --stories_dir narrativeqa/tmp
```

## Server control

```bash
# Require servers already running
python scripts/narrativeqa/run_all_experiments.py --no_auto_start_servers

# Override endpoints/ports
python scripts/narrativeqa/run_all_experiments.py \
  --vllm_endpoint http://127.0.0.1:8000/v1 \
  --embed_endpoint http://127.0.0.1:8001/v1
```
