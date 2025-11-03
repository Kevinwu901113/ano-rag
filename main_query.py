import argparse
import json

from query.query_processor import QueryProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured query runner")
    parser.add_argument("--question", required=True)
    parser.add_argument("--indexes_dir", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--lmstudio_endpoint", required=True)
    parser.add_argument("--lmstudio_model", required=True)
    args = parser.parse_args()

    qp = QueryProcessor(
        indexes_dir=args.indexes_dir,
        notes_path=args.notes,
        lmstudio_endpoint=args.lmstudio_endpoint,
        lmstudio_model=args.lmstudio_model,
    )
    result = qp.process(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
