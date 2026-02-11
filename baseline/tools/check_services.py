#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class Service:
    name: str
    base_url: str
    expected_model: str
    api_key: str


def list_models(service: Service) -> list[str]:
    client = OpenAI(base_url=service.base_url, api_key=service.api_key)
    models = client.models.list().data
    return [m.id for m in models]


def probe_embedding_dim(base_url: str, model: str, api_key: str) -> int:
    client = OpenAI(base_url=base_url, api_key=api_key)
    vec = client.embeddings.create(model=model, input="hello").data[0].embedding
    return len(vec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check baseline model services")
    parser.add_argument("--chat-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--embed-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--chat-model", default="qwen3-30b-a3b")
    parser.add_argument("--embed-model", default="qwen3-embedding")
    parser.add_argument("--expected-embedding-dim", type=int, default=4096)
    args = parser.parse_args()

    chat = Service("vLLM-chat", args.chat_url, args.chat_model, "EMPTY")
    embed = Service("vLLM-embed", args.embed_url, args.embed_model, "EMPTY")

    report: dict[str, object] = {"ok": True, "checks": {}}

    for svc in (chat, embed):
        models = list_models(svc)
        ok = svc.expected_model in models
        report["checks"][svc.name] = {
            "base_url": svc.base_url,
            "expected_model": svc.expected_model,
            "models": models,
            "ok": ok,
        }
        if not ok:
            report["ok"] = False

    dim = probe_embedding_dim(args.embed_url, args.embed_model, "EMPTY")
    dim_ok = dim == args.expected_embedding_dim
    report["checks"]["embedding_dim"] = {
        "model": args.embed_model,
        "observed": dim,
        "expected": args.expected_embedding_dim,
        "ok": dim_ok,
    }
    if not dim_ok:
        report["ok"] = False

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
