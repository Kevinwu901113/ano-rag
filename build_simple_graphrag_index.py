import argparse
import json
from typing import Dict
from loguru import logger
from structrag.llm_client import LLMChatClient
from config.config_loader import config as global_config
from baselines.simple_graphrag.build_graph import GraphBuilder

def main():
    parser = argparse.ArgumentParser(description="Build Simple GraphRAG Index")
    parser.add_argument("--doc_pool", type=str, required=True, help="Path to doc_pool.json or similar")
    parser.add_argument("--output_graph", type=str, default="simple_graphrag_graph.pkl", help="Output path for graph pickle")
    parser.add_argument("--output_chunks", type=str, default="simple_graphrag_chunk_store.pkl", help="Output path for chunk store pickle")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of docs to process (0 for all)")
    args = parser.parse_args()

    # 1. Initialize LLM Client
    lm_cfg = global_config.get("lmstudio", {})
    endpoint = lm_cfg.get("endpoint")
    model = lm_cfg.get("model")
    if not endpoint or not model:
        logger.error("LM Studio endpoint/model must be configured in config.yaml")
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
                    if "doc_id" in item and "text" in item:
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
    builder.build(docs)

    # 4. Save
    builder.save(args.output_graph, args.output_chunks)
    logger.info("Done.")

if __name__ == "__main__":
    main()
