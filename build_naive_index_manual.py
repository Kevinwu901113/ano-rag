import argparse
from baselines.naive_rag.index import MirageNaiveIndexer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-pool-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--embed-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Embedding device preference (auto prefers CUDA, falls back to CPU)",
    )
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--embed-max-length", type=int, default=512)
    parser.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-8B")
    norm = parser.add_mutually_exclusive_group()
    norm.add_argument("--embed-normalize", dest="embed_normalize", action="store_true")
    norm.add_argument("--no-embed-normalize", dest="embed_normalize", action="store_false")
    parser.set_defaults(embed_normalize=True)
    args = parser.parse_args()
    
    indexer = MirageNaiveIndexer()
    indexer.build(
        args.doc_pool_path,
        args.output_dir,
        embed_model=args.embed_model,
        embed_device=args.embed_device,
        embed_batch_size=args.embed_batch_size,
        embed_max_length=args.embed_max_length,
        embed_normalize=args.embed_normalize,
    )

if __name__ == "__main__":
    main()
