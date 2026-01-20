# RelRAG Core

Minimal, self-contained structured RAG core with build -> retrieve -> answer.

## Quickstart (CLI)

Build:

```bash
python -m relrag.cli.build \
  --input relrag/tests/fixtures/docs.jsonl \
  --out_dir /tmp/relrag_out \
  --endpoint http://127.0.0.1:8000/v1 \
  --model qwen3-30b-a3b
```

Retrieve:

```bash
python -m relrag.cli.retrieve \
  --question "Who wrote Python?" \
  --index /tmp/relrag_out/indexes \
  --notes /tmp/relrag_out/notes.jsonl > /tmp/relrag_result.json
```

Answer:

```bash
python -m relrag.cli.answer \
  --question "Who wrote Python?" \
  --evidences /tmp/relrag_result.json \
  --endpoint http://127.0.0.1:8000/v1 \
  --model qwen3-30b-a3b
```

## Config

- Default config file: `relrag/config/config.yaml`
- Override config path: set `ANO_RAG_CONFIG=/path/to/config.yaml`
- Override embedding model: set `RAG_EMBED_MODEL=...`

## LLM Endpoint/Model Override

By default, the LLM client normalizes to the built-in endpoint/model.
To allow custom endpoint/model (for testing or alternative servers), set:

```bash
RELRAG_ALLOW_CUSTOM_LLM=1
```
