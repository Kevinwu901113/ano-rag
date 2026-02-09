#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relrag.schema.schema_learner import SchemaLearner

def main():
    parser = argparse.ArgumentParser(description="Auto-build schema (attributes, vocab)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input notes JSONL")
    parser.add_argument("--mode", type=str, choices=["all", "attributes", "vocab"], default="all", help="What to build")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
        
    learner = SchemaLearner()
    
    if args.mode in ["all", "attributes"]:
        logger.info("Starting Attribute Learning...")
        learner.learn_attributes(input_path)
        
    if args.mode in ["all", "vocab"]:
        logger.info("Starting Vocab Learning...")
        learner.learn_vocab(input_path)

if __name__ == "__main__":
    main()
