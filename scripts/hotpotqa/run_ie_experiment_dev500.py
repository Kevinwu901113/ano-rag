#!/usr/bin/env python3
import argparse
import csv
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_DATA = "data/hotpot_dev_distractor_500_jsonl.jsonl"
DEFAULT_CONFIG = "relrag/config/config.yaml"
DEFAULT_OUTPUT_ROOT = "result/hotpot_ie_dev500"
DEFAULT_K_LIST = "1,2,3,5,10,20,30,50"


def _run_cmd(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}"
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{message}")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["method", "k", "IE", "Noise", "SNR", "n", "std", "ci_low", "ci_high", "run_dir"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _draw_line(pixels: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy), 1)
    for i in range(steps + 1):
        x = int(round(x0 + dx * (i / steps)))
        y = int(round(y0 + dy * (i / steps)))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        idx = (y * width + x) * 3
        pixels[idx : idx + 3] = bytes(color)


def _write_simple_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride : (y + 1) * stride])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    path.write_bytes(png)


def _plot_all_fallback(path: Path, grouped_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    width, height = 1020, 620
    pixels = bytearray([255] * (width * height * 3))
    left, right = 80, width - 25
    top, bottom = 35, height - 55
    _draw_line(pixels, width, height, left, bottom, right, bottom, (0, 0, 0))
    _draw_line(pixels, width, height, left, top, left, bottom, (0, 0, 0))

    all_k: List[int] = []
    for rows in grouped_rows.values():
        for row in rows:
            all_k.append(int(row["k"]))
    if not all_k:
        _write_simple_png(path, width, height, pixels)
        return
    k_min, k_max = min(all_k), max(all_k)
    if k_min == k_max:
        k_max = k_min + 1

    palette: List[Tuple[int, int, int]] = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
    ]

    def _x(k: int) -> int:
        return int(round(left + (right - left) * ((k - k_min) / float(k_max - k_min))))

    def _y(v: float) -> int:
        vv = max(0.0, min(1.0, float(v)))
        return int(round(bottom - (bottom - top) * vv))

    for idx, method in enumerate(sorted(grouped_rows.keys())):
        ordered = sorted(grouped_rows[method], key=lambda row: int(row["k"]))
        ks = [int(row["k"]) for row in ordered]
        ies = [float(row["IE"]) for row in ordered]
        color = palette[idx % len(palette)]
        for i in range(1, len(ks)):
            _draw_line(pixels, width, height, _x(ks[i - 1]), _y(ies[i - 1]), _x(ks[i]), _y(ies[i]), color)
        for k, ie in zip(ks, ies):
            cx, cy = _x(k), _y(ie)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    _draw_line(pixels, width, height, cx + dx, cy + dy, cx + dx, cy + dy, color)

    _write_simple_png(path, width, height, pixels)


def _plot_all(path: Path, grouped_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=170)
        for method, rows in grouped_rows.items():
            ordered = sorted(rows, key=lambda row: int(row["k"]))
            ks = [int(row["k"]) for row in ordered]
            ies = [float(row["IE"]) for row in ordered]
            lows = [float(row["ci_low"]) for row in ordered]
            highs = [float(row["ci_high"]) for row in ordered]
            ax.plot(ks, ies, marker="o", linewidth=2.0, label=method)
            ax.fill_between(ks, lows, highs, alpha=0.15)

        ax.set_xlabel("K")
        ax.set_ylabel("IE@K")
        ax.set_title("HotpotQA Dev500 IE Curves")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return
    except Exception:
        _plot_all_fallback(path, grouped_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HotpotQA dev500 IE@K experiments and produce summary curves.")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="hotpot_entry config")
    parser.add_argument("--data", default=DEFAULT_DATA, help="HotpotQA dev500 jsonl")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT, help="Output root for all methods")
    parser.add_argument("--cache_dir", help="Retriever cache dir (default: <output_root>/cache)")
    parser.add_argument("--reader", default="vllm", help="Reader passed to hotpot_entry")
    parser.add_argument("--workers", type=int, default=1, help="Workers per run")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit")
    parser.add_argument("--top_k", type=int, default=50, help="Retrieval top_k (Kmax)")
    parser.add_argument("--top_k_raw", type=int, help="Optional raw top_k")
    parser.add_argument("--overfetch", type=float, default=2.0, help="Overfetch multiplier")
    parser.add_argument("--min_overfetch", type=float, default=2.0, help="Minimum overfetch multiplier")
    parser.add_argument("--backfill_max_overfetch", type=float, default=4.0, help="Backfill max overfetch")
    parser.add_argument("--backfill_step", type=float, default=1.5, help="Backfill step")
    parser.add_argument("--backfill_rounds", type=int, default=3, help="Backfill rounds")
    parser.add_argument("--export_top_k_max", type=int, default=50, help="Exported ranked candidates cap")
    parser.add_argument("--predicate_random_seed", type=int, default=2026, help="Random seed for predicate mode random")
    parser.add_argument("--standard_retriever", default="hybrid", choices=["bm25", "dense", "hybrid"], help="Standard RAG baseline retriever")
    parser.add_argument("--include_random_predicate", action="store_true", help="Also run RelRAG with random predicate")
    parser.add_argument("--skip_run", action="store_true", help="Skip retrieval runs and only evaluate existing run dirs")
    parser.add_argument("--k_list", default=DEFAULT_K_LIST, help="K list for IE curve")
    parser.add_argument("--bootstrap_samples", type=int, default=1000, help="Bootstrap samples")
    parser.add_argument("--bootstrap_seed", type=int, default=2026, help="Bootstrap seed")
    parser.add_argument("--audit_samples", type=int, default=20, help="Title-system audit sample count per method")
    parser.add_argument("--force_build", action="store_true", help="Force rebuild indexes")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    data_path = (repo_root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    if args.cache_dir:
        cache_dir = (repo_root / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir).resolve()
    else:
        cache_dir = output_root / "cache"
    summary_dir = output_root / "summary"
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    methods: List[Dict[str, Any]] = [
        {
            "name": "standard_rag",
            "retriever": args.standard_retriever,
            "predicate_mode": "on",
        },
        {
            "name": "relrag_pred_on",
            "retriever": "hybrid",
            "predicate_mode": "on",
        },
        {
            "name": "relrag_wo_predicate",
            "retriever": "hybrid",
            "predicate_mode": "off",
        },
    ]
    if args.include_random_predicate:
        methods.append(
            {
                "name": "relrag_random_predicate",
                "retriever": "hybrid",
                "predicate_mode": "random",
            }
        )

    for method in methods:
        run_dir = output_root / method["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_run:
            cmd = [
                args.python,
                str(repo_root / "hotpot_entry.py"),
                "--config",
                str(config_path),
                "--data",
                str(data_path),
                "--cache_dir",
                str(cache_dir),
                "--output_dir",
                str(run_dir),
                "--reader",
                str(args.reader),
                "--retriever",
                str(method["retriever"]),
                "--top_k",
                str(int(args.top_k)),
                "--export_top_k_max",
                str(int(args.export_top_k_max)),
                "--retrieval_only",
                "true",
                "--disable_retriever_llm",
                "true",
                "--predicate_mode",
                str(method["predicate_mode"]),
                "--predicate_random_seed",
                str(int(args.predicate_random_seed)),
                "--workers",
                str(int(args.workers)),
                "--overfetch",
                str(float(args.overfetch)),
                "--min_overfetch",
                str(float(args.min_overfetch)),
                "--backfill_max_overfetch",
                str(float(args.backfill_max_overfetch)),
                "--backfill_step",
                str(float(args.backfill_step)),
                "--backfill_rounds",
                str(int(args.backfill_rounds)),
            ]
            if args.limit > 0:
                cmd.extend(["--limit", str(int(args.limit))])
            if args.top_k_raw is not None:
                cmd.extend(["--top_k_raw", str(int(args.top_k_raw))])
            if args.force_build:
                cmd.append("--force_build")
            print(f"[run] {method['name']}")
            print("[cmd]", " ".join(cmd))
            _run_cmd(cmd, cwd=repo_root)

        eval_cmd = [
            args.python,
            str(repo_root / "scripts" / "hotpotqa" / "eval_ie_curve.py"),
            "--data",
            str(data_path),
            "--run_dir",
            str(run_dir),
            "--method",
            str(method["name"]),
            "--out_dir",
            str(run_dir),
            "--k_list",
            str(args.k_list),
            "--bootstrap_samples",
            str(int(args.bootstrap_samples)),
            "--bootstrap_seed",
            str(int(args.bootstrap_seed)),
            "--audit_samples",
            str(int(args.audit_samples)),
        ]
        print(f"[eval] {method['name']}")
        print("[cmd]", " ".join(eval_cmd))
        _run_cmd(eval_cmd, cwd=repo_root)

    all_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    postprocess_signatures: Dict[str, Any] = {}
    retrieval_stats_by_method: Dict[str, Any] = {}
    for method in methods:
        run_dir = output_root / method["name"]
        csv_path = run_dir / "ie_curve.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing ie curve csv: {csv_path}")
        rows = _read_csv(csv_path)
        for row in rows:
            row = dict(row)
            row["run_dir"] = str(run_dir)
            all_rows.append(row)
            grouped.setdefault(str(row["method"]), []).append(row)
        run_meta = _read_optional_json(run_dir / "run_meta.json")
        postprocess_signatures[method["name"]] = run_meta.get("postprocess_signature")
        retrieval_stats_by_method[method["name"]] = run_meta.get("retrieval_stats")

    signature_values = [json.dumps(sig, sort_keys=True, ensure_ascii=False) for sig in postprocess_signatures.values() if sig]
    postprocess_signature_consistent = len(set(signature_values)) <= 1 if signature_values else False
    if not postprocess_signature_consistent:
        print("[warn] postprocess_signature differs across methods; check summary json")

    csv_all_path = summary_dir / "ie_curve_all.csv"
    json_all_path = summary_dir / "ie_curve_all.json"
    png_all_path = summary_dir / "ie_curve_all.png"
    _write_csv(csv_all_path, all_rows)
    json_all_path.write_text(
        json.dumps(
            {
                "data_path": str(data_path),
                "methods": methods,
                "cache_dir": str(cache_dir),
                "rows": all_rows,
                "postprocess_signature_consistent": postprocess_signature_consistent,
                "postprocess_signatures": postprocess_signatures,
                "retrieval_stats_by_method": retrieval_stats_by_method,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_all(png_all_path, grouped)

    print(f"[summary] wrote {csv_all_path}")
    print(f"[summary] wrote {json_all_path}")
    print(f"[summary] wrote {png_all_path}")


if __name__ == "__main__":
    main()
