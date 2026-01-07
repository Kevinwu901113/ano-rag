import argparse
import sys
import json
from relrag.api import retrieve
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Retrieve from RelRAG index")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--index", required=True, help="Index directory")
    parser.add_argument("--notes", required=True, help="Notes JSONL file")
    
    args = parser.parse_args()
    
    try:
        result = retrieve(
            question=args.question,
            index_dir=args.index,
            notes_path=args.notes
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
