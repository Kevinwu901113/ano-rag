#!/usr/bin/env python3
import argparse
from pathlib import Path
from loguru import logger
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relrag.schema.predicate_learner import PredicateLearner

def main():
    parser = argparse.ArgumentParser(description="Auto-build predicates.json")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input notes JSONL")
    parser.add_argument("--min-freq", type=int, default=5, help="Min frequency to cluster")
    
    args = parser.parse_args()
    
    learner = PredicateLearner()
    learner.learn_and_update(Path(args.input), min_freq=args.min_freq)

if __name__ == "__main__":
    main()
