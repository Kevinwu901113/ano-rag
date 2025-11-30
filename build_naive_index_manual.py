import argparse
from baselines.naive_rag.index import MirageNaiveIndexer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-pool-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    indexer = MirageNaiveIndexer()
    indexer.build(args.doc_pool_path, args.output_dir)

if __name__ == "__main__":
    main()
