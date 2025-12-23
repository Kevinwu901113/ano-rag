import argparse
import json
import asyncio
from typing import Dict
from loguru import logger
from structrag.llm_client import LLMChatClient
from config.config_loader import config as global_config
from baselines.simple_graphrag.build_graph import GraphBuilder

async def main():
    parser = argparse.ArgumentParser(description="Build Simple GraphRAG Index")
    parser.add_argument("--doc_pool", type=str, required=True, help="Path to doc_pool.json or similar")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save graph and chunk store")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of docs to process (0 for all)")
    parser.add_argument("--concurrency", type=int, default=50, help="Number of concurrent LLM requests")
    args = parser.parse_args()

    # 1. Initialize LLM Client
    lm_cfg = global_config.get("vllm", {})
    endpoint = lm_cfg.get("endpoint")
    model = lm_cfg.get("model")
    if not endpoint or not model:
        logger.error("vLLM endpoint/model must be configured in config.yaml")
        return

    llm_client = LLMChatClient(endpoint=endpoint, model=model, temperature=0.0)
    logger.info(f"Initialized LLM client with model: {model}")

    # 2. Load Documents
    docs: Dict[str, str] = {}
    logger.info(f"Loading documents from {args.doc_pool}...")
    try:
        with open(args.doc_pool, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
            # Handle different doc pool formats
            if isinstance(raw_data, dict):
                # Assume {doc_id: doc_obj} or {doc_id: text}
                for k, v in raw_data.items():
                    if isinstance(v, str):
                        docs[k] = v
                    elif isinstance(v, dict) and "text" in v:
                        docs[k] = v["text"]
                    elif isinstance(v, dict) and "content" in v:
                        docs[k] = v["content"]
            elif isinstance(raw_data, list):
                # Assume list of doc objects
                for item in raw_data:
                    if "mapped_id" in item and "doc_chunk" in item:
                        # MIRAGE format: use mapped_id or combine with doc_name if needed
                        # To ensure uniqueness if mapped_id is repeated for chunks, we might need a composite key
                        # But mapped_id seems to be the document ID.
                        # However, looking at the data, multiple entries have the same mapped_id but different chunks.
                        # It seems the input JSON is already chunked or has multiple parts.
                        # Let's use a unique key for each entry to avoid overwriting.
                        # We can use mapped_id + index
                        key = f"{item['mapped_id']}_{len(docs)}"
                        text = item.get("doc_name", "") + "\n" + item["doc_chunk"]
                        docs[key] = text
                    elif "doc_id" in item and "text" in item:
                        docs[item["doc_id"]] = item["text"]
                    elif "id" in item and "content" in item:
                        docs[item["id"]] = item["content"]
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        return

    if args.limit > 0:
        docs = dict(list(docs.items())[:args.limit])
    
    logger.info(f"Loaded {len(docs)} documents.")

    # 3. Build Graph
    builder = GraphBuilder(llm_client)
    await builder.build(docs, concurrency=args.concurrency)

    # 4. Save
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    output_graph = os.path.join(args.output_dir, "simple_graphrag_graph.pkl")
    output_chunks = os.path.join(args.output_dir, "simple_graphrag_chunk_store.pkl")
    
    builder.save(output_graph, output_chunks)
    logger.info(f"Saved graph to {output_graph} and chunks to {output_chunks}")

if __name__ == "__main__":
    asyncio.run(main())
