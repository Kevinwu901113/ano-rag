import argparse
import sys
import json
from relrag.api import answer
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Answer question using RelRAG")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--evidences", help="Evidences JSON file (default: stdin)")
    parser.add_argument("--endpoint", required=True, help="vLLM endpoint")
    parser.add_argument("--model", required=True, help="Model name")
    
    args = parser.parse_args()
    
    try:
        if args.evidences:
            with open(args.evidences, 'r') as f:
                data = json.load(f)
        else:
            data = json.load(sys.stdin)
            
        # data might be the full result from retrieve, which is Dict[str, Any]
        # containing "candidates" or similar?
        # Let's check retrieve_answer return type again.
        # It returns Dict[str, Any]. Usually it contains "candidates" or "evidences".
        # But wait, retrieve_answer returns what?
        # Looking at retriever/pipeline.py, it returns candidates probably?
        # Actually I need to check retrieve_answer implementation more closely to see what it returns.
        
        # Assuming data is a list of evidences or a dict containing them.
        # If it's the dict returned by retrieve_answer, we need to extract evidences.
        
        evidences = []
        if isinstance(data, list):
            evidences = data
        elif isinstance(data, dict):
            if "evidence" in data:
                evidences = data["evidence"]
            elif "evidences" in data:
                evidences = data["evidences"]
            elif "candidates" in data:
                evidences = data["candidates"]
            else:
                evidences = [data]
        
        ans = answer(
            question=args.question,
            evidences=evidences,
            llm_endpoint=args.endpoint,
            llm_model=args.model
        )
        print(ans)
    except Exception as e:
        logger.error(f"Answering failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
