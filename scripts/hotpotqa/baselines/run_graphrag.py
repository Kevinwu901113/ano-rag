import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
import re

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class MiniGraphRAG:
    def __init__(self, llm: LLMChatClient):
        self.llm = llm
        
    def build_and_query(self, context_data: List[List[Any]], question: str) -> str:
        """
        1. Extract entities/relations from 10 paragraphs.
        2. Build a mini text-based graph.
        3. Answer.
        """
        if len(context_data) > 10:
             context_data = context_data[:10]
             
        # Just concat for extraction to save calls? Or extract per paragraph.
        # To be safe and high quality, extract per paragraph but batching is hard here.
        # Let's do a single extraction pass if text fits, or per-paragraph.
        # Hotpot paragraphs are short.
        
        triples = []
        
        # Simplified: Concat all text, then extract graph (might be too long for extraction prompt output)
        # Better: Extract from each paragraph
        
        for title, sentences in context_data:
            text = "".join(sentences)
            # Extract
            prompt = f"""Extract knowledge triples (Subject, Relation, Object) from the text. Return as JSON list.
Text: {text}
JSON:"""
            try:
                resp = self.llm.chat([{"role": "user", "content": prompt}])
                # Heuristic parsing
                try:
                    # Try to find JSON list
                    match = re.search(r'\[.*\]', resp, re.DOTALL)
                    if match:
                        extracted = json.loads(match.group(0))
                        if isinstance(extracted, list):
                            triples.extend(extracted)
                except:
                    pass
            except:
                pass
                
        # Format graph
        graph_desc = "\n".join([f"{t.get('Subject', '')} -- {t.get('Relation', '')} --> {t.get('Object', '')}" for t in triples if isinstance(t, dict)])
        
        # Also provide original text for grounding? GraphRAG usually uses community summaries.
        # Here we just use the graph + original text as fallback or combined.
        # Strict GraphRAG relies on the graph.
        
        full_text = "\n\n".join([f"{t}\n{''.join(s)}" for t, s in context_data])
        
        prompt = f"""Answer the question using the knowledge graph and text below.
        
Graph:
{graph_desc[:4000]} 

Original Text:
{full_text[:4000]}

Question: {question}
Answer:"""

        return self.llm.chat([{"role": "user", "content": prompt}])

def main():
    parser = argparse.ArgumentParser(description="Run Mini GraphRAG on HotpotQA Distractor")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--limit", type=int, default=0)
    
    args = parser.parse_args()

    llm = LLMChatClient(endpoint=args.lm_endpoint, model=args.lm_model, temperature=0.0)
    graph_rag = MiniGraphRAG(llm)
    
    data = load_dataset(args.dataset)
    if args.limit > 0:
        data = data[:args.limit]
        
    predictions = {"answer": {}, "sp": {}}
    
    for item in tqdm(data):
        qid = item["_id"]
        question = item["question"]
        
        try:
            ans = graph_rag.build_and_query(item["context"], question)
            ans = ans.strip().replace("Answer:", "").strip()
            predictions["answer"][qid] = ans
            predictions["sp"][qid] = [] 
        except Exception as e:
            logger.error(f"Error Q {qid}: {e}")
            predictions["answer"][qid] = "error"
            
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)

if __name__ == "__main__":
    main()
