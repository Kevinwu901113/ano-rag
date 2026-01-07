import argparse
import sys
from relrag.api import build_index
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Build RelRAG index")
    parser.add_argument("--input", required=True, help="Input documents file or directory")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--endpoint", required=True, help="vLLM endpoint")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    
    args = parser.parse_args()
    
    try:
        stats = build_index(
            docs_input=args.input,
            output_dir=args.out_dir,
            llm_endpoint=args.endpoint,
            llm_model=args.model,
            temperature=args.temperature
        )
        print(f"Build complete. Stats: {stats}")
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
