import argparse
import json

from query.query_processor import QueryProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured query runner")
    parser.add_argument("--question", required=True)
    parser.add_argument("--indexes_dir", required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()

    qp = QueryProcessor(
        indexes_dir=args.indexes_dir,
        notes_path=args.notes,
    )
    result = qp.process(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
