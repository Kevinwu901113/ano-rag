#!/usr/bin/env python3
import argparse
import csv
import json
import random
import statistics
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 2026


def normalize_title(title: Any) -> str:
    text = str(title or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def _parse_k_list(raw: str) -> List[int]:
    values: List[int] = []
    seen = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("k_list is empty after parsing")
    return values


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_gold_titles(data_path: Path) -> Dict[str, Dict[str, set[str]]]:
    payload: Dict[str, Dict[str, set[str]]] = {}
    for row in _load_jsonl(data_path):
        qid = str(row.get("_id") or row.get("id") or "").strip()
        if not qid:
            continue
        exact: set[str] = set()
        lowered: set[str] = set()
        for item in row.get("supporting_facts") or []:
            if not isinstance(item, (list, tuple)) or len(item) < 1:
                continue
            title = normalize_title(item[0])
            if not title:
                continue
            exact.add(title)
            lowered.add(title.lower())
        payload[qid] = {"exact": exact, "lowered": lowered}
    return payload


def _load_ranked_titles(retrieval_path: Path) -> Dict[str, List[str]]:
    rows: Dict[str, List[Tuple[int, str]]] = {}
    for row in _load_jsonl(retrieval_path):
        qid = str(row.get("qid") or "").strip()
        if not qid:
            continue
        try:
            rank = int(row.get("rank"))
        except (TypeError, ValueError):
            continue
        title = normalize_title(row.get("doc_title"))
        rows.setdefault(qid, []).append((rank, title))
    ordered: Dict[str, List[str]] = {}
    for qid, values in rows.items():
        values.sort(key=lambda item: item[0])
        ordered[qid] = [title for _, title in values]
    return ordered


def _is_effective(title: str, gold_titles: Dict[str, set[str]]) -> int:
    normalized = normalize_title(title)
    if not normalized:
        return 0
    if normalized in gold_titles["exact"]:
        return 1
    if normalized.lower() in gold_titles["lowered"]:
        return 1
    return 0


def _compute_ie_samples(
    gold_by_qid: Dict[str, Dict[str, set[str]]],
    ranked_by_qid: Dict[str, List[str]],
    k_list: List[int],
) -> Dict[int, List[float]]:
    per_k_values: Dict[int, List[float]] = {k: [] for k in k_list}
    for qid in gold_by_qid.keys():
        gold_titles = gold_by_qid[qid]
        ranked = ranked_by_qid.get(qid, [])
        for k in k_list:
            hit_sum = 0
            for idx in range(k):
                if idx >= len(ranked):
                    continue
                hit_sum += _is_effective(ranked[idx], gold_titles)
            per_k_values[k].append(float(hit_sum) / float(k))
    return per_k_values


def _build_title_audit_rows(
    gold_by_qid: Dict[str, Dict[str, set[str]]],
    ranked_by_qid: Dict[str, List[str]],
    sample_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if sample_size <= 0:
        return rows
    qids = sorted(gold_by_qid.keys())[:sample_size]
    for qid in qids:
        gold_titles = sorted(gold_by_qid[qid]["exact"])
        ranked_titles = ranked_by_qid.get(qid, [])
        rows.append(
            {
                "qid": qid,
                "gold_titles": gold_titles,
                "top10_doc_titles": ranked_titles[:10],
            }
        )
    return rows


def _bootstrap_ci(values: List[float], samples: int, seed: int) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for i in range(samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(float(sum(sample) / float(n)))
    means.sort()
    lo_idx = int((len(means) - 1) * 0.025)
    hi_idx = int((len(means) - 1) * 0.975)
    return float(means[lo_idx]), float(means[hi_idx])


def _infer_method_name(run_dir: Path) -> str:
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            reader = str(meta.get("reader") or "").strip()
            retriever = str(meta.get("retriever") or "").strip()
            predicate_mode = str(meta.get("predicate_mode") or "").strip()
            if reader and retriever:
                if predicate_mode:
                    return f"{reader}_{retriever}_{predicate_mode}"
                return f"{reader}_{retriever}"
        except Exception:
            pass
    return run_dir.name


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["method", "k", "IE", "Noise", "SNR", "n", "std", "ci_low", "ci_high"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        row = pixels[y * stride : (y + 1) * stride]
        raw.extend(row)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    path.write_bytes(png)


def _plot_curve_fallback(path: Path, rows: List[Dict[str, Any]]) -> None:
    width, height = 900, 520
    pixels = bytearray([255] * (width * height * 3))

    left, right = 70, width - 25
    top, bottom = 30, height - 45
    _draw_line(pixels, width, height, left, bottom, right, bottom, (0, 0, 0))
    _draw_line(pixels, width, height, left, top, left, bottom, (0, 0, 0))

    ordered = sorted(rows, key=lambda row: int(row["k"]))
    ks = [int(row["k"]) for row in ordered]
    ies = [float(row["IE"]) for row in ordered]
    if not ks:
        _write_simple_png(path, width, height, pixels)
        return
    k_min = min(ks)
    k_max = max(ks)
    if k_min == k_max:
        k_max = k_min + 1

    def _x(k: int) -> int:
        return int(round(left + (right - left) * ((k - k_min) / float(k_max - k_min))))

    def _y(v: float) -> int:
        vv = max(0.0, min(1.0, float(v)))
        return int(round(bottom - (bottom - top) * vv))

    for i in range(1, len(ks)):
        _draw_line(pixels, width, height, _x(ks[i - 1]), _y(ies[i - 1]), _x(ks[i]), _y(ies[i]), (31, 119, 180))
    for k, ie in zip(ks, ies):
        cx, cy = _x(k), _y(ie)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                _draw_line(pixels, width, height, cx + dx, cy + dy, cx + dx, cy + dy, (31, 119, 180))

    _write_simple_png(path, width, height, pixels)


def _plot_curve(path: Path, method: str, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = [int(row["k"]) for row in rows]
        ies = [float(row["IE"]) for row in rows]
        ci_low = [float(row["ci_low"]) for row in rows]
        ci_high = [float(row["ci_high"]) for row in rows]

        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
        ax.plot(ks, ies, marker="o", linewidth=2.0, label=method)
        ax.fill_between(ks, ci_low, ci_high, alpha=0.2)
        ax.set_xlabel("K")
        ax.set_ylabel("IE@K")
        ax.set_title(f"HotpotQA IE Curve - {method}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return
    except Exception:
        _plot_curve_fallback(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IE@K/Noise@K/SNR@K from exported final_top50 retrieval results.")
    parser.add_argument("--data", required=True, help="HotpotQA jsonl path (e.g. dev500)")
    parser.add_argument("--run_dir", required=True, help="Run directory containing retrieval/final_top50.jsonl")
    parser.add_argument("--method", help="Method name override")
    parser.add_argument("--retrieval_file", help="Optional retrieval file path override")
    parser.add_argument("--out_dir", help="Optional output dir (default: run_dir)")
    parser.add_argument("--k_list", default="1,2,3,5,10,20,30,50", help="Comma-separated K list")
    parser.add_argument("--bootstrap_samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap_seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--audit_samples", type=int, default=0, help="Write title-system audit rows (gold titles vs top10 doc titles)")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    k_list = _parse_k_list(args.k_list)

    retrieval_file = Path(args.retrieval_file).resolve() if args.retrieval_file else (run_dir / "retrieval" / "final_top50.jsonl")
    if not retrieval_file.exists():
        raise FileNotFoundError(f"retrieval file not found: {retrieval_file}")
    if not data_path.exists():
        raise FileNotFoundError(f"data file not found: {data_path}")

    method = str(args.method or _infer_method_name(run_dir))
    gold_by_qid = _load_gold_titles(data_path)
    ranked_by_qid = _load_ranked_titles(retrieval_file)
    ie_samples = _compute_ie_samples(gold_by_qid, ranked_by_qid, k_list)
    n = len(gold_by_qid)

    rows: List[Dict[str, Any]] = []
    for k in k_list:
        values = ie_samples[k]
        ie = float(sum(values) / float(len(values))) if values else 0.0
        noise = float(1.0 - ie)
        snr = float(ie / (noise + 1e-9))
        std = float(statistics.pstdev(values)) if values else 0.0
        ci_low, ci_high = _bootstrap_ci(values, int(args.bootstrap_samples), int(args.bootstrap_seed) + int(k))
        rows.append(
            {
                "method": method,
                "k": int(k),
                "IE": ie,
                "Noise": noise,
                "SNR": snr,
                "n": int(n),
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    csv_path = out_dir / "ie_curve.csv"
    json_path = out_dir / "ie_curve.json"
    png_path = out_dir / "ie_curve.png"

    _write_csv(csv_path, rows)
    _write_json(
        json_path,
        {
            "method": method,
            "run_dir": str(run_dir),
            "data_path": str(data_path),
            "retrieval_file": str(retrieval_file),
            "k_list": k_list,
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.bootstrap_seed),
            "n": int(n),
            "rows": rows,
        },
    )
    _plot_curve(png_path, method, rows)

    if int(args.audit_samples) > 0:
        audit_rows = _build_title_audit_rows(gold_by_qid, ranked_by_qid, int(args.audit_samples))
        audit_path = out_dir / "title_audit_samples.json"
        _write_json(audit_path, {"sample_count": len(audit_rows), "rows": audit_rows})
        print(f"[ie] wrote {audit_path}")

    print(f"[ie] method={method} n={n}")
    print(f"[ie] wrote {csv_path}")
    print(f"[ie] wrote {json_path}")
    print(f"[ie] wrote {png_path}")


if __name__ == "__main__":
    main()
