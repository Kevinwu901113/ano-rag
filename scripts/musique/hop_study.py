#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency in local env
    yaml = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
HOP_RE = re.compile(r"^(?P<hop>\d+)hop(?:\d+)?__", re.IGNORECASE)
RUN_NAME_RE = re.compile(r"^(?:run_)?(?P<method>[a-zA-Z0-9_]+)_hop(?P<hop>\d+)_seed(?P<seed>-?\d+)$")


DEFAULT_METHODS = ("standard", "relrag")
SUPPORTED_METHODS = {"standard", "relrag", "relrag_no_walk"}
DEFAULT_HOPS = (2, 3, 4)
DEFAULT_SEEDS = (11, 29, 47)


def _now_ts() -> int:
    return int(time.time())


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_csv_ints(value: str) -> List[int]:
    items: List[int] = []
    for token in re.split(r"[,\s]+", str(value or "").strip()):
        if not token:
            continue
        items.append(int(token))
    return items


def _parse_csv_strings(value: str) -> List[str]:
    items: List[str] = []
    for token in re.split(r"[,\s]+", str(value or "").strip()):
        token = token.strip()
        if token:
            items.append(token)
    return items


def _allocate_total_balanced(
    available_by_hop: Dict[int, int],
    hops: Sequence[int],
    total_samples: int,
) -> Tuple[Dict[int, int], int, int]:
    allocation: Dict[int, int] = {int(h): 0 for h in hops}
    total_available = sum(max(0, int(available_by_hop.get(int(h), 0))) for h in hops)
    if total_samples <= 0 or total_available <= 0:
        return allocation, 0, total_available

    target = min(int(total_samples), int(total_available))
    remaining = target
    active = [int(h) for h in hops if int(available_by_hop.get(int(h), 0)) > 0]
    if not active:
        return allocation, 0, total_available

    idx = 0
    while remaining > 0 and active:
        if idx >= len(active):
            idx = 0
        hop = active[idx]
        if allocation[hop] < int(available_by_hop.get(hop, 0)):
            allocation[hop] += 1
            remaining -= 1
            idx += 1
            continue
        active.pop(idx)

    effective = target - remaining
    return allocation, effective, total_available


def _allocate_total_proportional(
    available_by_hop: Dict[int, int],
    hops: Sequence[int],
    total_samples: int,
) -> Tuple[Dict[int, int], int, int]:
    allocation: Dict[int, int] = {int(h): 0 for h in hops}
    total_available = sum(max(0, int(available_by_hop.get(int(h), 0))) for h in hops)
    if total_samples <= 0 or total_available <= 0:
        return allocation, 0, total_available

    target = min(int(total_samples), int(total_available))
    raw_quota: Dict[int, float] = {}
    base: Dict[int, int] = {}
    for hop in hops:
        h = int(hop)
        avail = max(0, int(available_by_hop.get(h, 0)))
        quota = (float(avail) * float(target)) / float(total_available) if total_available > 0 else 0.0
        raw_quota[h] = quota
        base[h] = min(avail, int(math.floor(quota)))
    allocation.update(base)

    remainder = target - sum(allocation.values())
    if remainder > 0:
        # Largest remainder method.
        order = sorted(
            [int(h) for h in hops],
            key=lambda h: (raw_quota[h] - float(base[h]), raw_quota[h], -h),
            reverse=True,
        )
        idx = 0
        guard = 0
        while remainder > 0 and guard < 100000:
            guard += 1
            if idx >= len(order):
                idx = 0
            h = order[idx]
            cap = max(0, int(available_by_hop.get(h, 0)))
            if allocation[h] < cap:
                allocation[h] += 1
                remainder -= 1
            idx += 1

    effective = target - max(0, remainder)
    return allocation, effective, total_available


def _infer_hop(record: Dict[str, Any]) -> Tuple[int, str]:
    explicit = record.get("hop")
    if isinstance(explicit, int) and explicit > 0:
        return int(explicit), "field"
    qid = str(record.get("id") or record.get("_id") or "")
    match = HOP_RE.match(qid)
    if match:
        return int(match.group("hop")), "id_prefix"
    decomp = record.get("question_decomposition")
    if isinstance(decomp, list) and decomp:
        return len(decomp), "decomposition_len"
    raise ValueError(f"Cannot infer hop for record id={qid!r}")


def _normalize_support_facts(raw: Any) -> List[Tuple[str, int]]:
    if not isinstance(raw, list):
        return []
    seen = set()
    facts: List[Tuple[str, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = str(item[0] or "").strip()
        if not title:
            continue
        try:
            idx = int(item[1])
        except (TypeError, ValueError):
            continue
        key = (title, idx)
        if key in seen:
            continue
        seen.add(key)
        facts.append(key)
    return facts


def _support_prf(gold: Sequence[Tuple[str, int]], pred: Sequence[Tuple[str, int]]) -> Tuple[float, float, float]:
    gold_set = set(gold)
    pred_set = set(pred)
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gold_set or not pred_set:
        return 0.0, 0.0, 0.0
    common = len(gold_set.intersection(pred_set))
    if common <= 0:
        return 0.0, 0.0, 0.0
    precision = common / float(len(pred_set))
    recall = common / float(len(gold_set))
    f1 = (2.0 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / float(len(values)))


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    ci: float,
    seed: int,
) -> Tuple[float, float, float]:
    if not values:
        nan = float("nan")
        return nan, nan, nan
    n = len(values)
    values_list = [float(v) for v in values]
    avg = _mean(values_list)
    if n == 1:
        return avg, avg, avg
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(max(200, int(samples))):
        total = 0.0
        for _ in range(n):
            total += values_list[rng.randrange(n)]
        draws.append(total / float(n))
    draws.sort()
    alpha = max(0.0, min(1.0, 1.0 - float(ci)))
    low_idx = int((alpha / 2.0) * len(draws))
    high_idx = int((1.0 - alpha / 2.0) * len(draws)) - 1
    low_idx = max(0, min(low_idx, len(draws) - 1))
    high_idx = max(0, min(high_idx, len(draws) - 1))
    return avg, draws[low_idx], draws[high_idx]


def _paired_bootstrap_diff(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int,
    ci: float,
    seed: int,
) -> Dict[str, Any]:
    n = min(len(left), len(right))
    if n <= 0:
        return {
            "n": 0,
            "mean_left": float("nan"),
            "mean_right": float("nan"),
            "mean_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
        }
    diffs = [float(left[i]) - float(right[i]) for i in range(n)]
    mean_left = _mean(left[:n])
    mean_right = _mean(right[:n])
    mean_diff = _mean(diffs)
    if n == 1:
        return {
            "n": n,
            "mean_left": mean_left,
            "mean_right": mean_right,
            "mean_diff": mean_diff,
            "ci_low": mean_diff,
            "ci_high": mean_diff,
            "p_value": 1.0,
        }
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(max(200, int(samples))):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        draws.append(total / float(n))
    draws.sort()
    alpha = max(0.0, min(1.0, 1.0 - float(ci)))
    low_idx = int((alpha / 2.0) * len(draws))
    high_idx = int((1.0 - alpha / 2.0) * len(draws)) - 1
    low_idx = max(0, min(low_idx, len(draws) - 1))
    high_idx = max(0, min(high_idx, len(draws) - 1))
    less_or_equal_zero = sum(1 for v in draws if v <= 0.0) / float(len(draws))
    greater_or_equal_zero = sum(1 for v in draws if v >= 0.0) / float(len(draws))
    p_value = min(1.0, 2.0 * min(less_or_equal_zero, greater_or_equal_zero))
    return {
        "n": n,
        "mean_left": mean_left,
        "mean_right": mean_right,
        "mean_diff": mean_diff,
        "ci_low": draws[low_idx],
        "ci_high": draws[high_idx],
        "p_value": p_value,
    }


def _fit_exp(points: Sequence[Tuple[int, float]], eps: float = 1e-8) -> Dict[str, Any]:
    filtered = [(int(h), float(v)) for h, v in points if float(v) > 0.0]
    if len(filtered) < 2:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan"), "n": len(filtered)}
    xs = [float(h) for h, _ in filtered]
    ys = [math.log(v + eps) for _, v in filtered]
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx <= 0:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan"), "n": len(filtered)}
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    preds = [intercept + slope * x for x in xs]
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "a": math.exp(intercept),
        "b": -slope,
        "r2": r2,
        "n": len(filtered),
    }


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return len(str(text).split())


def _hop_sort_key(hop: int) -> Tuple[int, int]:
    if hop in {2, 3, 4}:
        return (0, hop)
    return (1, hop)


def _method_label(method: str) -> str:
    return {
        "standard": "Standard RAG",
        "relrag": "RelRAG (walk)",
        "relrag_no_walk": "RelRAG (no walk)",
    }.get(method, method)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _format_f(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(number):
        return "nan"
    return f"{number:.{digits}f}"


def _render_simple_svg_curve(
    *,
    out_path: Path,
    title: str,
    y_label: str,
    series: Dict[str, Dict[int, Tuple[float, float, float]]],
    hops: Sequence[int],
) -> None:
    width = 860
    height = 520
    left = 90
    right = 40
    top = 70
    bottom = 90
    chart_w = width - left - right
    chart_h = height - top - bottom

    all_upper: List[float] = []
    for method_values in series.values():
        for hop in hops:
            if hop in method_values:
                all_upper.append(method_values[hop][2])
    y_max = max(all_upper) if all_upper else 1.0
    y_max = min(1.0, max(0.1, y_max + 0.05))

    hops_sorted = sorted(hops, key=_hop_sort_key)
    if not hops_sorted:
        return
    x_min = float(min(hops_sorted))
    x_max = float(max(hops_sorted))
    if x_max <= x_min:
        x_max = x_min + 1.0

    def x_of(h: int) -> float:
        ratio = (float(h) - x_min) / (x_max - x_min)
        return left + ratio * chart_w

    def y_of(v: float) -> float:
        vv = max(0.0, min(y_max, float(v)))
        ratio = vv / y_max if y_max > 0 else 0.0
        return top + (1.0 - ratio) * chart_h

    palette = {
        "standard": "#1f77b4",
        "relrag": "#d62728",
        "relrag_no_walk": "#2ca02c",
    }
    lines: List[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{width/2:.1f}" y="34" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>')

    for tick in range(0, 6):
        v = y_max * tick / 5.0
        y = y_of(v)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e8e8e8" stroke-width="1"/>')
        lines.append(
            f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#555">{v:.2f}</text>'
        )

    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333" stroke-width="1.5"/>')
    lines.append(
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="{left-56}" y="{top + chart_h/2:.2f}" transform="rotate(-90 {left-56:.2f},{top + chart_h/2:.2f})" text-anchor="middle" font-size="14" font-family="Arial">{y_label}</text>'
    )
    lines.append(
        f'<text x="{left + chart_w/2:.2f}" y="{height-34}" text-anchor="middle" font-size="14" font-family="Arial">Hop</text>'
    )

    for hop in hops_sorted:
        x = x_of(hop)
        lines.append(f'<line x1="{x:.2f}" y1="{height-bottom}" x2="{x:.2f}" y2="{height-bottom+6}" stroke="#333" stroke-width="1"/>')
        lines.append(
            f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" font-size="13" font-family="Arial">{hop}</text>'
        )

    legend_x = left + 16
    legend_y = top - 20
    legend_gap = 26
    for idx, method in enumerate(sorted(series.keys())):
        color = palette.get(method, "#444")
        y = legend_y + idx * legend_gap
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<circle cx="{legend_x+12}" cy="{y}" r="4" fill="{color}"/>')
        lines.append(
            f'<text x="{legend_x+34}" y="{y+4}" text-anchor="start" font-size="13" font-family="Arial">{_method_label(method)}</text>'
        )

    for method, method_values in sorted(series.items()):
        color = palette.get(method, "#444")
        points: List[Tuple[float, float]] = []
        for hop in hops_sorted:
            if hop not in method_values:
                continue
            mean, low, high = method_values[hop]
            x = x_of(hop)
            y = y_of(mean)
            low_y = y_of(low)
            high_y = y_of(high)
            points.append((x, y))
            lines.append(f'<line x1="{x:.2f}" y1="{high_y:.2f}" x2="{x:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<line x1="{x-6:.2f}" y1="{high_y:.2f}" x2="{x+6:.2f}" y2="{high_y:.2f}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<line x1="{x-6:.2f}" y1="{low_y:.2f}" x2="{x+6:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}"/>')
        if len(points) >= 2:
            d = " ".join(
                [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
                + [f"L {x:.2f} {y:.2f}" for x, y in points[1:]]
            )
            lines.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>')

    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _deep_set_bool(cfg: Dict[str, Any], keys: Sequence[str], value: bool) -> None:
    cursor = cfg
    for key in keys[:-1]:
        node = cursor.get(key)
        if not isinstance(node, dict):
            node = {}
            cursor[key] = node
        cursor = node
    cursor[str(keys[-1])] = bool(value)


def _build_relrag_no_walk_config(base_config: Path, out_path: Path) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is required to generate relrag_no_walk config.")
    if not base_config.exists():
        raise FileNotFoundError(f"base config not found: {base_config}")
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        payload = {}
    _deep_set_bool(payload, ("retriever", "structured", "walk_enabled"), False)
    _deep_set_bool(payload, ("retriever", "structured", "multihop_rescue_enabled"), False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def _prepare_dataset(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hops = sorted(set(_parse_csv_ints(args.hops) if args.hops else list(DEFAULT_HOPS)))
    if not hops:
        raise ValueError("No hops specified.")
    if any(h <= 0 for h in hops):
        raise ValueError(f"Invalid hops: {hops}")

    tag = args.tag.strip() if args.tag else input_path.stem
    dedupe_by_id = bool(args.dedupe_id)
    answerable_only = bool(args.answerable_only)
    sample_per_hop = int(args.sample_per_hop or 0)
    total_samples = int(args.total_samples or 0)
    total_sample_strategy = str(args.total_sample_strategy or "balanced").strip().lower()
    if total_sample_strategy not in {"balanced", "proportional"}:
        raise ValueError("total_sample_strategy must be one of: balanced, proportional")
    sample_seed = int(args.sample_seed)

    by_hop: Dict[int, List[Dict[str, Any]]] = {hop: [] for hop in hops}
    seen_ids = set()
    dropped_duplicate = 0
    dropped_answerable = 0
    dropped_hop = 0
    hop_source_counter = Counter()

    for record in _read_jsonl(input_path):
        qid = str(record.get("id") or record.get("_id") or "")
        if dedupe_by_id and qid:
            if qid in seen_ids:
                dropped_duplicate += 1
                continue
            seen_ids.add(qid)
        if answerable_only and not bool(record.get("answerable", True)):
            dropped_answerable += 1
            continue
        hop, source = _infer_hop(record)
        if hop not in by_hop:
            dropped_hop += 1
            continue
        enriched = dict(record)
        enriched["hop"] = int(hop)
        enriched["hop_source"] = source
        by_hop[hop].append(enriched)
        hop_source_counter[source] += 1

    rng = random.Random(sample_seed)
    selected_by_hop: Dict[int, List[Dict[str, Any]]] = {}
    for hop in hops:
        rows = sorted(by_hop.get(hop, []), key=lambda r: str(r.get("id") or ""))
        if sample_per_hop > 0 and len(rows) > sample_per_hop:
            idxs = sorted(rng.sample(range(len(rows)), sample_per_hop))
            rows = [rows[i] for i in idxs]
        selected_by_hop[hop] = rows

    total_allocation: Dict[int, int] = {int(h): len(rows) for h, rows in selected_by_hop.items()}
    total_effective = sum(total_allocation.values())
    total_available_after_hop_cap = total_effective
    total_requested = total_samples
    if total_samples > 0:
        available_for_total = {int(h): len(rows) for h, rows in selected_by_hop.items()}
        if total_sample_strategy == "proportional":
            total_allocation, total_effective, total_available_after_hop_cap = _allocate_total_proportional(
                available_for_total,
                hops,
                total_samples,
            )
        else:
            total_allocation, total_effective, total_available_after_hop_cap = _allocate_total_balanced(
                available_for_total,
                hops,
                total_samples,
            )
        for hop in hops:
            rows = selected_by_hop.get(hop, [])
            keep_n = int(total_allocation.get(hop, 0))
            if keep_n <= 0:
                selected_by_hop[hop] = []
            elif keep_n < len(rows):
                idxs = sorted(rng.sample(range(len(rows)), keep_n))
                selected_by_hop[hop] = [rows[i] for i in idxs]
            else:
                selected_by_hop[hop] = rows

    hop_files: Dict[str, str] = {}
    hop_id_files: Dict[str, str] = {}
    stats_rows: List[Dict[str, Any]] = []
    combined: List[Dict[str, Any]] = []

    for hop in sorted(selected_by_hop.keys(), key=_hop_sort_key):
        rows = selected_by_hop[hop]
        combined.extend(rows)
        hop_jsonl = output_dir / f"{tag}_hop{hop}.jsonl"
        _write_jsonl(hop_jsonl, rows)
        hop_files[str(hop)] = str(hop_jsonl)

        ids = [str(row.get("id") or row.get("_id") or "") for row in rows]
        ids_path = output_dir / f"{tag}_hop{hop}_ids.json"
        ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
        hop_id_files[str(hop)] = str(ids_path)
        stats_rows.append({"hop": hop, "count": len(rows)})

    combined = sorted(combined, key=lambda r: (_to_int(r.get("hop"), 0), str(r.get("id") or "")))
    all_path = output_dir / f"{tag}_hop_all.jsonl"
    _write_jsonl(all_path, combined)

    stats_csv = output_dir / "hop_counts.csv"
    with stats_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["hop", "count"])
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)

    manifest = {
        "created_at": _now_ts(),
        "input": str(input_path),
        "tag": tag,
        "answerable_only": answerable_only,
        "dedupe_by_id": dedupe_by_id,
        "sample_per_hop": sample_per_hop,
        "total_samples_requested": total_requested,
        "total_sample_strategy": total_sample_strategy,
        "total_samples_effective": total_effective,
        "total_available_after_hop_cap": total_available_after_hop_cap,
        "total_allocation_by_hop": {str(h): int(total_allocation.get(h, 0)) for h in hops},
        "sample_seed": sample_seed,
        "hops": hops,
        "total_selected": len(combined),
        "dropped_duplicate": dropped_duplicate,
        "dropped_answerable": dropped_answerable,
        "dropped_hop": dropped_hop,
        "hop_source_counts": dict(hop_source_counter),
        "hop_files": hop_files,
        "hop_id_files": hop_id_files,
        "all_hops_file": str(all_path),
        "counts_csv": str(stats_csv),
    }
    _write_json(output_dir / "manifest.json", manifest)

    summary_lines = [
        "# MuSiQue Hop Dataset Manifest",
        "",
        f"- input: `{input_path}`",
        f"- answerable_only: `{answerable_only}`",
        f"- dedupe_by_id: `{dedupe_by_id}`",
        f"- sample_per_hop: `{sample_per_hop}`",
        f"- total_samples_requested: `{total_requested}`",
        f"- total_sample_strategy: `{total_sample_strategy}`",
        f"- total_samples_effective: `{total_effective}`",
        f"- sample_seed: `{sample_seed}`",
        f"- total_selected: `{len(combined)}`",
        f"- dropped_duplicate: `{dropped_duplicate}`",
        f"- dropped_answerable: `{dropped_answerable}`",
        f"- dropped_hop: `{dropped_hop}`",
        "",
        "| hop | count | jsonl | ids |",
        "| --- | ---: | --- | --- |",
    ]
    for hop in hops:
        summary_lines.append(
            f"| {hop} | {len(selected_by_hop.get(hop, []))} | `{hop_files.get(str(hop), '')}` | `{hop_id_files.get(str(hop), '')}` |"
        )
    (output_dir / "manifest.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[prepare] wrote {output_dir / 'manifest.json'}")


@dataclass
class RunDefinition:
    method: str
    hop: int
    seed: int
    run_name: str
    run_dir: Path
    data_path: Path
    script_path: Path
    cmd: List[str]
    env: Dict[str, str]
    reader: str
    retriever: str
    config_path: Optional[Path]


def _build_run_command(
    *,
    repo_root: Path,
    method: str,
    reader: str,
    retriever: str,
    script_path: Path,
    run_dir: Path,
    data_path: Path,
    split: str,
    workers: int,
    top_k: int,
    top_k_raw: int,
    min_overfetch: float,
    backfill_max_overfetch: float,
    backfill_step: float,
    backfill_rounds: int,
    llm_retry_on_empty: int,
    llm_retry_max_evidence: int,
    config_path: Optional[Path],
    endpoint: Optional[str],
    model: Optional[str],
    openai_model: Optional[str],
    openai_temperature: Optional[float],
    openai_max_tokens: Optional[int],
    include_decomposition_sp: bool,
    limit: int,
) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--run_dir",
        str(run_dir),
        "--data",
        str(data_path),
        "--split",
        str(split),
        "--reader",
        str(reader),
        "--retriever",
        str(retriever),
        "--workers",
        str(max(1, int(workers))),
        "--top_k",
        str(max(1, int(top_k))),
        "--top_k_raw",
        str(max(1, int(top_k_raw))),
        "--min_overfetch",
        str(float(min_overfetch)),
        "--backfill_max_overfetch",
        str(float(backfill_max_overfetch)),
        "--backfill_step",
        str(float(backfill_step)),
        "--backfill_rounds",
        str(max(0, int(backfill_rounds))),
        "--llm_retry_on_empty",
        str(max(0, int(llm_retry_on_empty))),
        "--llm_retry_max_evidence",
        str(max(0, int(llm_retry_max_evidence))),
    ]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    if endpoint:
        cmd.extend(["--endpoint", str(endpoint)])
    if model:
        cmd.extend(["--model", str(model)])
    if limit > 0:
        cmd.extend(["--limit", str(int(limit))])
    if include_decomposition_sp:
        cmd.append("--include_decomposition_sp")
    else:
        cmd.append("--no_decomposition_sp")
    if method in {"relrag", "relrag_no_walk"}:
        cmd.append("--no_structured_answer")
    if reader == "openai":
        if openai_model:
            cmd.extend(["--openai_model", str(openai_model)])
        if openai_temperature is not None:
            cmd.extend(["--openai_temperature", str(float(openai_temperature))])
        if openai_max_tokens is not None:
            cmd.extend(["--openai_max_tokens", str(int(openai_max_tokens))])
    return cmd


def _build_matrix(args: argparse.Namespace) -> None:
    data_manifest_path = Path(args.data_manifest).resolve()
    manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    hop_files = manifest.get("hop_files") or {}
    if not isinstance(hop_files, dict):
        raise ValueError(f"Invalid manifest hop_files: {data_manifest_path}")

    methods = _parse_csv_strings(args.methods) if args.methods else list(DEFAULT_METHODS)
    methods = [m.strip() for m in methods if m.strip()]
    if not methods:
        raise ValueError("No methods selected.")
    unknown = sorted(set(methods) - SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}; allowed={sorted(SUPPORTED_METHODS)}")

    hops = _parse_csv_ints(args.hops) if args.hops else list(DEFAULT_HOPS)
    seeds = _parse_csv_ints(args.seeds) if args.seeds else list(DEFAULT_SEEDS)
    if not hops:
        raise ValueError("No hops selected.")
    if not seeds:
        raise ValueError("No seeds selected.")

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT

    base_config = Path(args.config).resolve() if args.config else repo_root / "relrag" / "config" / "config.yaml"
    no_walk_config: Optional[Path] = None
    if "relrag_no_walk" in methods:
        no_walk_config = _build_relrag_no_walk_config(
            base_config,
            run_root / "configs" / "config.relrag_no_walk.yaml",
        )

    retriever_standard = args.standard_retriever.strip().lower()
    if retriever_standard not in {"bm25", "dense"}:
        raise ValueError("standard_retriever must be one of: bm25,dense")
    retriever_relrag = args.relrag_retriever.strip().lower()
    if retriever_relrag not in {"bm25", "dense", "hybrid", "structured"}:
        raise ValueError("relrag_retriever must be one of: bm25,dense,hybrid,structured")

    definitions: List[RunDefinition] = []
    for method in methods:
        for hop in sorted(set(hops), key=_hop_sort_key):
            hop_data = hop_files.get(str(hop))
            if not hop_data:
                raise FileNotFoundError(f"hop {hop} not found in {data_manifest_path}")
            data_path = Path(hop_data).resolve()
            if not data_path.exists():
                raise FileNotFoundError(f"hop dataset missing: {data_path}")
            for seed in sorted(set(seeds)):
                run_name = f"{method}_hop{hop}_seed{seed}"
                run_dir = run_root / f"run_{run_name}"
                if method == "standard":
                    script_path = repo_root / "musique_baseline_entry.py"
                    retriever = retriever_standard
                    config_path = base_config
                elif method == "relrag":
                    script_path = repo_root / "musique_entry.py"
                    retriever = retriever_relrag
                    config_path = base_config
                else:
                    script_path = repo_root / "musique_entry.py"
                    retriever = retriever_relrag
                    config_path = no_walk_config

                cmd = _build_run_command(
                    repo_root=repo_root,
                    method=method,
                    reader=args.reader,
                    retriever=retriever,
                    script_path=script_path,
                    run_dir=run_dir,
                    data_path=data_path,
                    split=args.split,
                    workers=args.workers,
                    top_k=args.top_k,
                    top_k_raw=args.top_k_raw,
                    min_overfetch=args.min_overfetch,
                    backfill_max_overfetch=args.backfill_max_overfetch,
                    backfill_step=args.backfill_step,
                    backfill_rounds=args.backfill_rounds,
                    llm_retry_on_empty=args.llm_retry_on_empty,
                    llm_retry_max_evidence=args.llm_retry_max_evidence,
                    config_path=config_path,
                    endpoint=args.endpoint,
                    model=args.model,
                    openai_model=args.openai_model,
                    openai_temperature=args.openai_temperature,
                    openai_max_tokens=args.openai_max_tokens,
                    include_decomposition_sp=bool(args.include_decomposition_sp),
                    limit=args.limit,
                )

                env = dict(os.environ)
                env["PYTHONHASHSEED"] = str(seed)
                env["ANO_HOP_SEED"] = str(seed)
                if args.emb_endpoint:
                    env["EMB_ENDPOINT"] = str(args.emb_endpoint)
                if args.endpoint and str(args.endpoint).rstrip("/") != "http://127.0.0.1:8000/v1":
                    env["RELRAG_ALLOW_CUSTOM_LLM"] = "1"
                definitions.append(
                    RunDefinition(
                        method=method,
                        hop=int(hop),
                        seed=int(seed),
                        run_name=run_name,
                        run_dir=run_dir,
                        data_path=data_path,
                        script_path=script_path,
                        cmd=cmd,
                        env=env,
                        reader=args.reader,
                        retriever=retriever,
                        config_path=config_path,
                    )
                )

    definitions = sorted(definitions, key=lambda d: (d.method, d.hop, d.seed))
    run_records: List[Dict[str, Any]] = []
    commands_lines: List[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for d in definitions:
        env_exports = []
        for key in ["PYTHONHASHSEED", "ANO_HOP_SEED", "EMB_ENDPOINT", "RELRAG_ALLOW_CUSTOM_LLM"]:
            if key in d.env:
                env_exports.append(f"{key}={shlex.quote(str(d.env[key]))}")
        cmd_str = " ".join(shlex.quote(part) for part in d.cmd)
        prefix = " ".join(env_exports)
        commands_lines.append(f"{prefix} {cmd_str}".strip())

    commands_path = run_root / "run_matrix.sh"
    commands_path.write_text("\n".join(commands_lines) + "\n", encoding="utf-8")
    os.chmod(commands_path, 0o755)

    for d in definitions:
        row: Dict[str, Any] = {
            "run_name": d.run_name,
            "method": d.method,
            "hop": d.hop,
            "seed": d.seed,
            "status": "planned",
            "run_dir": str(d.run_dir),
            "data_path": str(d.data_path),
            "script": str(d.script_path),
            "reader": d.reader,
            "retriever": d.retriever,
            "config_path": str(d.config_path) if d.config_path else None,
            "cmd": d.cmd,
        }
        if args.dry_run:
            if args.progress:
                print(f"[matrix] planned {row['run_name']} ({row['method']}, hop={row['hop']}, seed={row['seed']})")
            run_records.append(row)
            continue

        completed_path = d.run_dir / "completed.json"
        log_dir = d.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "driver.log"
        row["log_path"] = str(log_path)

        if args.resume and completed_path.exists():
            row["status"] = "skipped"
            if args.progress:
                print(f"[matrix] skip {row['run_name']} (completed.json exists)")
            run_records.append(row)
            continue

        run_idx = len(run_records) + 1
        total_runs = len(definitions)
        started = time.time()
        row["started_at"] = int(started)
        if args.progress:
            print(
                "[matrix] [{}/{}] start {} | method={} hop={} seed={} | log={}".format(
                    run_idx,
                    total_runs,
                    d.run_name,
                    d.method,
                    d.hop,
                    d.seed,
                    log_path,
                )
            )
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                d.cmd,
                cwd=str(repo_root),
                env=d.env,
                stdout=handle,
                stderr=handle,
            )
            heartbeat_sec = max(0.0, float(args.heartbeat_sec))
            next_heartbeat = started + heartbeat_sec if heartbeat_sec > 0 else float("inf")
            while True:
                code = proc.poll()
                if code is not None:
                    row["return_code"] = int(code)
                    break
                now = time.time()
                if args.progress and now >= next_heartbeat:
                    elapsed = now - started
                    print(
                        "[matrix] [{}/{}] running {} | elapsed={:.1f}s".format(
                            run_idx,
                            total_runs,
                            d.run_name,
                            elapsed,
                        )
                    )
                    next_heartbeat = now + heartbeat_sec
                time.sleep(0.5)
        ended = time.time()
        row["ended_at"] = int(ended)
        row["duration_sec"] = round(max(0.0, ended - started), 2)
        row["status"] = "ok" if int(row["return_code"]) == 0 else "failed"
        if args.progress:
            print(
                "[matrix] [{}/{}] done {} | status={} | duration={}s".format(
                    run_idx,
                    total_runs,
                    d.run_name,
                    row["status"],
                    row["duration_sec"],
                )
            )
        run_records.append(row)

    failed = [r for r in run_records if r.get("status") == "failed"]
    matrix_payload = {
        "created_at": _now_ts(),
        "repo_root": str(repo_root),
        "data_manifest": str(data_manifest_path),
        "run_root": str(run_root),
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "reader": args.reader,
        "methods": methods,
        "hops": sorted(set(hops)),
        "seeds": sorted(set(seeds)),
        "controls": {
            "top_k": int(args.top_k),
            "top_k_raw": int(args.top_k_raw),
            "min_overfetch": float(args.min_overfetch),
            "backfill_max_overfetch": float(args.backfill_max_overfetch),
            "backfill_step": float(args.backfill_step),
            "backfill_rounds": int(args.backfill_rounds),
            "llm_retry_on_empty": int(args.llm_retry_on_empty),
            "llm_retry_max_evidence": int(args.llm_retry_max_evidence),
            "workers": int(args.workers),
            "split": args.split,
            "include_decomposition_sp": bool(args.include_decomposition_sp),
            "standard_retriever": retriever_standard,
            "relrag_retriever": retriever_relrag,
            "config": str(base_config),
            "relrag_no_walk_config": str(no_walk_config) if no_walk_config else None,
            "endpoint": args.endpoint,
            "model": args.model,
            "emb_endpoint": args.emb_endpoint,
            "openai_model": args.openai_model,
            "openai_temperature": args.openai_temperature,
            "openai_max_tokens": args.openai_max_tokens,
        },
        "commands_path": str(commands_path),
        "runs": run_records,
    }
    _write_json(run_root / "matrix.json", matrix_payload)

    md_lines = [
        "# MuSiQue Hop Experiment Matrix",
        "",
        f"- run_root: `{run_root}`",
        f"- dry_run: `{bool(args.dry_run)}`",
        f"- commands: `{commands_path}`",
        "",
        "| run_name | method | hop | seed | status | retriever | run_dir |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in run_records:
        md_lines.append(
            f"| {row.get('run_name')} | {row.get('method')} | {row.get('hop')} | {row.get('seed')} | {row.get('status')} | {row.get('retriever')} | `{row.get('run_dir')}` |"
        )
    if failed:
        md_lines.append("")
        md_lines.append("## Failed")
        for row in failed:
            md_lines.append(f"- {row.get('run_name')} (return_code={row.get('return_code')})")
    (run_root / "matrix.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[matrix] wrote {run_root / 'matrix.json'}")
    print(f"[matrix] wrote {commands_path}")
    if failed:
        print(f"[matrix] failed runs: {len(failed)}")
        sys.exit(1)


def _discover_runs(run_root: Path) -> List[Dict[str, Any]]:
    matrix_path = run_root / "matrix.json"
    if matrix_path.exists():
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        runs = payload.get("runs") or []
        if isinstance(runs, list):
            discovered = []
            for row in runs:
                if not isinstance(row, dict):
                    continue
                run_dir = row.get("run_dir")
                run_name = row.get("run_name")
                if not run_dir or not run_name:
                    continue
                discovered.append(
                    {
                        "run_name": str(run_name),
                        "run_dir": Path(str(run_dir)),
                        "method": row.get("method"),
                        "hop": row.get("hop"),
                        "seed": row.get("seed"),
                    }
                )
            if discovered:
                return discovered
    discovered = []
    for run_dir in sorted(run_root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        match = RUN_NAME_RE.match(run_dir.name)
        if not match:
            continue
        discovered.append(
            {
                "run_name": run_dir.name[len("run_") :] if run_dir.name.startswith("run_") else run_dir.name,
                "run_dir": run_dir,
                "method": match.group("method"),
                "hop": int(match.group("hop")),
                "seed": int(match.group("seed")),
            }
        )
    return discovered


def _extract_row_identity(row: Dict[str, Any], fallback: Dict[str, Any]) -> Tuple[str, int, int]:
    method = str(fallback.get("method") or "unknown")
    hop = _to_int(fallback.get("hop"), 0)
    seed = _to_int(fallback.get("seed"), 0)
    run_name = str(fallback.get("run_name") or "")
    match = RUN_NAME_RE.match(run_name)
    if match:
        method = match.group("method")
        hop = int(match.group("hop"))
        seed = int(match.group("seed"))
    return method, hop, seed


def _analyze_runs(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(run_root)
    if not runs:
        raise RuntimeError(f"No runs discovered under {run_root}")

    per_question_rows: List[Dict[str, Any]] = []
    run_level_rows: List[Dict[str, Any]] = []

    for run in runs:
        run_dir = Path(run["run_dir"]).resolve()
        pred_path = run_dir / "predictions.jsonl"
        if not pred_path.exists():
            continue
        run_meta_path = run_dir / "run_meta.json"
        cfg_path = run_dir / "config.resolved.json"
        run_meta = {}
        if run_meta_path.exists():
            try:
                run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            except Exception:
                run_meta = {}
        cfg_hash = None
        if cfg_path.exists():
            cfg_hash = hashlib.sha1(cfg_path.read_bytes()).hexdigest()

        method, run_hop, seed = _extract_row_identity(run, run)
        metric_lists: Dict[str, List[float]] = defaultdict(list)
        n_rows = 0

        for rec in _read_jsonl(pred_path):
            qid = str(rec.get("id") or rec.get("_id") or "")
            try:
                qhop, _ = _infer_hop(rec)
            except Exception:
                qhop = run_hop
            if run_hop and qhop and run_hop != qhop:
                qhop = run_hop

            answer_f1 = _to_float((rec.get("metrics") or {}).get("f1"), 0.0)
            answer_em = _to_float((rec.get("metrics") or {}).get("em"), 0.0)
            gold_sp = _normalize_support_facts(rec.get("gold_sp") or rec.get("supporting_facts") or [])
            pred_sp = _normalize_support_facts(rec.get("pred_sp") or rec.get("sp") or [])
            sp_prec, sp_recall, sp_f1 = _support_prf(gold_sp, pred_sp)
            joint_coverage = 1.0 if set(gold_sp).issubset(set(pred_sp)) else 0.0

            retrieved_raw = rec.get("retrieved_context_raw") or []
            retrieved_topk = rec.get("retrieved_context_topk") or rec.get("retrieved_context") or []
            raw_len = len(retrieved_raw) if isinstance(retrieved_raw, list) else 0
            topk_len = len(retrieved_topk) if isinstance(retrieved_topk, list) else 0
            context_token_est = 0
            if isinstance(retrieved_topk, list):
                for item in retrieved_topk:
                    if not isinstance(item, dict):
                        continue
                    txt = str(item.get("text") or item.get("evidence") or "")
                    context_token_est += _estimate_text_tokens(txt)

            metric_lists["answer_f1"].append(answer_f1)
            metric_lists["answer_em"].append(answer_em)
            metric_lists["support_f1"].append(sp_f1)
            metric_lists["joint_coverage"].append(joint_coverage)
            metric_lists["top_k_final"].append(_to_float(rec.get("top_k_final"), 0.0))
            metric_lists["top_k_raw"].append(_to_float(rec.get("top_k_raw"), 0.0))
            metric_lists["duplicate_rate"].append(_to_float(rec.get("duplicate_rate"), 0.0))
            metric_lists["context_token_est"].append(float(context_token_est))
            metric_lists["chain_len"].append(float(len(gold_sp)))

            row = {
                "run_name": run.get("run_name"),
                "run_dir": str(run_dir),
                "method": method,
                "hop": int(qhop),
                "seed": int(seed),
                "id": qid,
                "question": rec.get("question"),
                "prediction": rec.get("answer") or rec.get("short_answer"),
                "gold_answer": rec.get("gold_answer"),
                "answer_f1": answer_f1,
                "answer_em": answer_em,
                "support_precision": sp_prec,
                "support_recall": sp_recall,
                "support_f1": sp_f1,
                "joint_coverage": joint_coverage,
                "gold_support_size": len(gold_sp),
                "pred_support_size": len(pred_sp),
                "top_k": _to_int(rec.get("top_k"), 0),
                "top_k_raw": _to_int(rec.get("top_k_raw"), 0),
                "top_k_final": _to_int(rec.get("top_k_final"), 0),
                "duplicate_rate": _to_float(rec.get("duplicate_rate"), 0.0),
                "overfetch_factor": _to_float(rec.get("overfetch_factor"), 0.0),
                "retrieved_raw_count": raw_len,
                "retrieved_topk_count": topk_len,
                "context_token_est": context_token_est,
                "chain_len": len(gold_sp),
                "fallback_reason": rec.get("fallback_reason"),
                "answer_source": rec.get("answer_source"),
                "run_meta_path": str(run_meta_path) if run_meta_path.exists() else "",
                "config_resolved_path": str(cfg_path) if cfg_path.exists() else "",
                "config_hash": cfg_hash,
                "run_meta": run_meta,
                "retrieved_context_raw": retrieved_raw if isinstance(retrieved_raw, list) else [],
                "retrieved_context_topk": retrieved_topk if isinstance(retrieved_topk, list) else [],
                "final_evidence": ((rec.get("intermediate") or {}).get("retrieve_result") or {}).get("evidence")
                if isinstance(rec.get("intermediate"), dict)
                else None,
            }
            per_question_rows.append(row)
            n_rows += 1

        if n_rows <= 0:
            continue
        run_level_rows.append(
            {
                "run_name": run.get("run_name"),
                "run_dir": str(run_dir),
                "method": method,
                "hop": int(run_hop),
                "seed": int(seed),
                "n": n_rows,
                "answer_f1_mean": _mean(metric_lists["answer_f1"]),
                "support_f1_mean": _mean(metric_lists["support_f1"]),
                "joint_coverage_mean": _mean(metric_lists["joint_coverage"]),
                "top_k_final_mean": _mean(metric_lists["top_k_final"]),
                "top_k_raw_mean": _mean(metric_lists["top_k_raw"]),
                "duplicate_rate_mean": _mean(metric_lists["duplicate_rate"]),
                "context_token_est_mean": _mean(metric_lists["context_token_est"]),
                "chain_len_mean": _mean(metric_lists["chain_len"]),
            }
        )

    if not per_question_rows:
        raise RuntimeError("No prediction rows loaded for analysis.")

    audit_jsonl = output_dir / "per_question_audit.jsonl"
    _write_jsonl(audit_jsonl, per_question_rows)
    audit_csv = output_dir / "per_question_audit.csv"
    scalar_fields = [
        "run_name",
        "run_dir",
        "method",
        "hop",
        "seed",
        "id",
        "prediction",
        "gold_answer",
        "answer_f1",
        "answer_em",
        "support_precision",
        "support_recall",
        "support_f1",
        "joint_coverage",
        "gold_support_size",
        "pred_support_size",
        "top_k",
        "top_k_raw",
        "top_k_final",
        "duplicate_rate",
        "overfetch_factor",
        "retrieved_raw_count",
        "retrieved_topk_count",
        "context_token_est",
        "chain_len",
        "fallback_reason",
        "answer_source",
        "run_meta_path",
        "config_resolved_path",
        "config_hash",
    ]
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for row in per_question_rows:
            writer.writerow({field: row.get(field) for field in scalar_fields})

    run_summary_csv = output_dir / "run_level_summary.csv"
    run_fields = [
        "run_name",
        "run_dir",
        "method",
        "hop",
        "seed",
        "n",
        "answer_f1_mean",
        "support_f1_mean",
        "joint_coverage_mean",
        "top_k_final_mean",
        "top_k_raw_mean",
        "duplicate_rate_mean",
        "context_token_est_mean",
        "chain_len_mean",
    ]
    with run_summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        for row in sorted(run_level_rows, key=lambda r: (str(r.get("method")), int(r.get("hop", 0)), int(r.get("seed", 0)))):
            writer.writerow(row)

    grouped: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    key_counter: Dict[Tuple[str, int], set] = defaultdict(set)
    for row in per_question_rows:
        key = (str(row["method"]), int(row["hop"]))
        grouped[key]["answer_f1"].append(_to_float(row["answer_f1"], 0.0))
        grouped[key]["support_f1"].append(_to_float(row["support_f1"], 0.0))
        grouped[key]["joint_coverage"].append(_to_float(row["joint_coverage"], 0.0))
        grouped[key]["top_k_final"].append(_to_float(row["top_k_final"], 0.0))
        grouped[key]["top_k_raw"].append(_to_float(row["top_k_raw"], 0.0))
        grouped[key]["duplicate_rate"].append(_to_float(row["duplicate_rate"], 0.0))
        grouped[key]["context_token_est"].append(_to_float(row["context_token_est"], 0.0))
        grouped[key]["chain_len"].append(_to_float(row["chain_len"], 0.0))
        key_counter[key].add((int(row["seed"]), str(row["id"])))

    summary_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda x: (x[0], _hop_sort_key(x[1]))):
        method, hop = key
        metrics = grouped[key]
        answer_mean, answer_low, answer_high = _bootstrap_mean_ci(
            metrics["answer_f1"],
            samples=args.bootstrap_samples,
            ci=args.ci,
            seed=args.bootstrap_seed + hop,
        )
        support_mean, support_low, support_high = _bootstrap_mean_ci(
            metrics["support_f1"],
            samples=args.bootstrap_samples,
            ci=args.ci,
            seed=args.bootstrap_seed + 17 + hop,
        )
        joint_mean, joint_low, joint_high = _bootstrap_mean_ci(
            metrics["joint_coverage"],
            samples=args.bootstrap_samples,
            ci=args.ci,
            seed=args.bootstrap_seed + 37 + hop,
        )
        summary_rows.append(
            {
                "method": method,
                "hop": hop,
                "n_samples": len(metrics["answer_f1"]),
                "n_pairs": len(key_counter[key]),
                "answer_f1_mean": answer_mean,
                "answer_f1_ci_low": answer_low,
                "answer_f1_ci_high": answer_high,
                "support_f1_mean": support_mean,
                "support_f1_ci_low": support_low,
                "support_f1_ci_high": support_high,
                "joint_coverage_mean": joint_mean,
                "joint_coverage_ci_low": joint_low,
                "joint_coverage_ci_high": joint_high,
                "top_k_final_mean": _mean(metrics["top_k_final"]),
                "top_k_raw_mean": _mean(metrics["top_k_raw"]),
                "duplicate_rate_mean": _mean(metrics["duplicate_rate"]),
                "context_token_est_mean": _mean(metrics["context_token_est"]),
                "chain_len_mean": _mean(metrics["chain_len"]),
            }
        )

    summary_csv = output_dir / "summary_by_hop.csv"
    if summary_rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in sorted(summary_rows, key=lambda r: (str(r["method"]), _hop_sort_key(int(r["hop"])))):
                writer.writerow(row)

    by_method_hop: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    for row in summary_rows:
        by_method_hop[str(row["method"])][int(row["hop"])] = row

    decay_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    for method, hop_map in sorted(by_method_hop.items()):
        if 2 in hop_map and 4 in hop_map:
            f2 = _to_float(hop_map[2]["answer_f1_mean"], float("nan"))
            f4 = _to_float(hop_map[4]["answer_f1_mean"], float("nan"))
            c2 = _to_float(hop_map[2]["joint_coverage_mean"], float("nan"))
            c4 = _to_float(hop_map[4]["joint_coverage_mean"], float("nan"))
            decay_rows.append(
                {
                    "method": method,
                    "answer_drop_2_to_4": f2 - f4 if not (math.isnan(f2) or math.isnan(f4)) else float("nan"),
                    "answer_ratio_4_over_2": (f4 / f2) if f2 > 0 else float("nan"),
                    "coverage_drop_2_to_4": c2 - c4 if not (math.isnan(c2) or math.isnan(c4)) else float("nan"),
                    "coverage_ratio_4_over_2": (c4 / c2) if c2 > 0 else float("nan"),
                }
            )

        points_answer = [(hop, _to_float(row["answer_f1_mean"], 0.0)) for hop, row in sorted(hop_map.items(), key=lambda x: x[0])]
        fit_answer = _fit_exp(points_answer)
        fit_rows.append({"method": method, "metric": "answer_f1", **fit_answer})

        points_cov = [(hop, _to_float(row["joint_coverage_mean"], 0.0)) for hop, row in sorted(hop_map.items(), key=lambda x: x[0])]
        fit_cov = _fit_exp(points_cov)
        fit_rows.append({"method": method, "metric": "joint_coverage", **fit_cov})

    if decay_rows:
        with (output_dir / "decay_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(decay_rows[0].keys()))
            writer.writeheader()
            for row in decay_rows:
                writer.writerow(row)

    if fit_rows:
        with (output_dir / "exp_fit_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fit_rows[0].keys()))
            writer.writeheader()
            for row in fit_rows:
                writer.writerow(row)

    condition_rows: List[Dict[str, Any]] = []
    cond_groups: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    for row in per_question_rows:
        cond_key = (str(row["method"]), int(row["hop"]), int(_to_int(row["joint_coverage"], 0)))
        cond_groups[cond_key].append(_to_float(row["answer_f1"], 0.0))
    for key, vals in sorted(cond_groups.items(), key=lambda x: (x[0][0], _hop_sort_key(x[0][1]), x[0][2])):
        method, hop, covered = key
        condition_rows.append(
            {
                "method": method,
                "hop": hop,
                "joint_coverage": covered,
                "n": len(vals),
                "answer_f1_mean": _mean(vals),
                "answer_f1_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
        )
    if condition_rows:
        with (output_dir / "conditional_answer_f1_by_joint_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0].keys()))
            writer.writeheader()
            for row in condition_rows:
                writer.writerow(row)

    method_pairs = []
    preferred_pairs = [("relrag", "standard"), ("relrag", "relrag_no_walk"), ("relrag_no_walk", "standard")]
    methods_available = sorted({str(row["method"]) for row in per_question_rows})
    for left, right in preferred_pairs:
        if left in methods_available and right in methods_available:
            method_pairs.append((left, right))

    paired_rows: List[Dict[str, Any]] = []
    for left, right in method_pairs:
        for hop in sorted({int(row["hop"]) for row in per_question_rows}, key=_hop_sort_key):
            left_map: Dict[Tuple[int, str], Dict[str, float]] = {}
            right_map: Dict[Tuple[int, str], Dict[str, float]] = {}
            for row in per_question_rows:
                if int(row["hop"]) != hop:
                    continue
                key = (int(row["seed"]), str(row["id"]))
                if row["method"] == left:
                    left_map[key] = {
                        "answer_f1": _to_float(row["answer_f1"], 0.0),
                        "joint_coverage": _to_float(row["joint_coverage"], 0.0),
                        "support_f1": _to_float(row["support_f1"], 0.0),
                    }
                elif row["method"] == right:
                    right_map[key] = {
                        "answer_f1": _to_float(row["answer_f1"], 0.0),
                        "joint_coverage": _to_float(row["joint_coverage"], 0.0),
                        "support_f1": _to_float(row["support_f1"], 0.0),
                    }
            shared_keys = sorted(set(left_map.keys()).intersection(set(right_map.keys())))
            if not shared_keys:
                continue
            for metric in ("answer_f1", "joint_coverage", "support_f1"):
                left_vals = [left_map[key][metric] for key in shared_keys]
                right_vals = [right_map[key][metric] for key in shared_keys]
                stats_payload = _paired_bootstrap_diff(
                    left_vals,
                    right_vals,
                    samples=args.bootstrap_samples,
                    ci=args.ci,
                    seed=args.bootstrap_seed + hop * 13 + len(metric),
                )
                paired_rows.append(
                    {
                        "left_method": left,
                        "right_method": right,
                        "hop": hop,
                        "metric": metric,
                        **stats_payload,
                    }
                )
    if paired_rows:
        with (output_dir / "paired_bootstrap.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
            writer.writeheader()
            for row in paired_rows:
                writer.writerow(row)

    answer_series: Dict[str, Dict[int, Tuple[float, float, float]]] = defaultdict(dict)
    coverage_series: Dict[str, Dict[int, Tuple[float, float, float]]] = defaultdict(dict)
    for row in summary_rows:
        method = str(row["method"])
        hop = int(row["hop"])
        answer_series[method][hop] = (
            _to_float(row["answer_f1_mean"], 0.0),
            _to_float(row["answer_f1_ci_low"], 0.0),
            _to_float(row["answer_f1_ci_high"], 0.0),
        )
        coverage_series[method][hop] = (
            _to_float(row["joint_coverage_mean"], 0.0),
            _to_float(row["joint_coverage_ci_low"], 0.0),
            _to_float(row["joint_coverage_ci_high"], 0.0),
        )

    hops_for_plot = sorted({int(row["hop"]) for row in summary_rows}, key=_hop_sort_key)
    _render_simple_svg_curve(
        out_path=output_dir / "answer_f1_vs_hop.svg",
        title="Answer F1 vs Hop (95% CI)",
        y_label="Answer F1",
        series=answer_series,
        hops=hops_for_plot,
    )
    _render_simple_svg_curve(
        out_path=output_dir / "joint_coverage_vs_hop.svg",
        title="Joint Evidence Coverage vs Hop (95% CI)",
        y_label="Joint Coverage",
        series=coverage_series,
        hops=hops_for_plot,
    )

    summary_md_lines = [
        "# MuSiQue Hop Study Summary",
        "",
        f"- run_root: `{run_root}`",
        f"- analyzed_runs: `{len(run_level_rows)}`",
        f"- per_question_rows: `{len(per_question_rows)}`",
        f"- bootstrap_samples: `{args.bootstrap_samples}`",
        f"- ci: `{args.ci}`",
        "",
        "## Main Table",
        "",
        "| method | hop | n | Answer F1 (CI) | Support F1 (CI) | Joint Coverage (CI) | top_k_final |",
        "| --- | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in sorted(summary_rows, key=lambda r: (str(r["method"]), _hop_sort_key(int(r["hop"])))):
        summary_md_lines.append(
            "| {method} | {hop} | {n} | {af1} [{al}, {ah}] | {sf1} [{sl}, {sh}] | {jc} [{jl}, {jh}] | {tk} |".format(
                method=row["method"],
                hop=row["hop"],
                n=row["n_samples"],
                af1=_format_f(row["answer_f1_mean"]),
                al=_format_f(row["answer_f1_ci_low"]),
                ah=_format_f(row["answer_f1_ci_high"]),
                sf1=_format_f(row["support_f1_mean"]),
                sl=_format_f(row["support_f1_ci_low"]),
                sh=_format_f(row["support_f1_ci_high"]),
                jc=_format_f(row["joint_coverage_mean"]),
                jl=_format_f(row["joint_coverage_ci_low"]),
                jh=_format_f(row["joint_coverage_ci_high"]),
                tk=_format_f(row["top_k_final_mean"], digits=2),
            )
        )

    summary_md_lines.extend(
        [
            "",
            "## Decay",
            "",
            "| method | Answer Drop 2->4 | Answer Ratio 4/2 | Coverage Drop 2->4 | Coverage Ratio 4/2 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in decay_rows:
        summary_md_lines.append(
            f"| {row['method']} | {_format_f(row['answer_drop_2_to_4'])} | {_format_f(row['answer_ratio_4_over_2'])} | {_format_f(row['coverage_drop_2_to_4'])} | {_format_f(row['coverage_ratio_4_over_2'])} |"
        )

    if paired_rows:
        summary_md_lines.extend(
            [
                "",
                "## Paired Bootstrap",
                "",
                "| left | right | hop | metric | mean_diff | 95% CI | p_value | n |",
                "| --- | --- | ---: | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for row in paired_rows:
            summary_md_lines.append(
                "| {left} | {right} | {hop} | {metric} | {diff} | [{low}, {high}] | {p} | {n} |".format(
                    left=row["left_method"],
                    right=row["right_method"],
                    hop=row["hop"],
                    metric=row["metric"],
                    diff=_format_f(row["mean_diff"]),
                    low=_format_f(row["ci_low"]),
                    high=_format_f(row["ci_high"]),
                    p=_format_f(row["p_value"]),
                    n=row["n"],
                )
            )

    summary_md_lines.extend(
        [
            "",
            "## Plots",
            "",
            f"- `answer_f1_vs_hop.svg`",
            f"- `joint_coverage_vs_hop.svg`",
            "",
            "## Artifacts",
            "",
            f"- per-question audit: `{audit_jsonl}`",
            f"- per-question csv: `{audit_csv}`",
            f"- run-level summary: `{run_summary_csv}`",
            f"- grouped summary: `{summary_csv}`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary_md_lines) + "\n", encoding="utf-8")

    print(f"[analyze] wrote {output_dir / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuSiQue hop-stratified experiment toolkit (prepare / matrix / analyze)."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = subparsers.add_parser("prepare", help="Build hop-labeled MuSiQue datasets and ID manifests.")
    p_prepare.add_argument("--input", required=True, help="Input MuSiQue JSONL path.")
    p_prepare.add_argument("--output_dir", required=True, help="Output directory for manifests and jsonl files.")
    p_prepare.add_argument("--tag", default="", help="Filename prefix tag (default: input stem).")
    p_prepare.add_argument("--hops", default="2,3,4", help="Hop values to keep, e.g. 2,3,4")
    p_prepare.add_argument("--answerable_only", action="store_true", help="Keep only answerable=true rows.")
    p_prepare.add_argument("--include_unanswerable", action="store_false", dest="answerable_only", help="Keep both answerable and unanswerable rows.")
    p_prepare.set_defaults(answerable_only=True)
    p_prepare.add_argument("--dedupe_id", action="store_true", help="Drop duplicated ids, keep first occurrence.")
    p_prepare.add_argument("--no_dedupe_id", action="store_false", dest="dedupe_id", help="Keep duplicate ids.")
    p_prepare.set_defaults(dedupe_id=True)
    p_prepare.add_argument("--sample_per_hop", type=int, default=0, help="Optional cap per hop (0 disables).")
    p_prepare.add_argument("--total_samples", type=int, default=0, help="Optional total sample cap across all hops (0 disables).")
    p_prepare.add_argument(
        "--total_sample_strategy",
        default="balanced",
        choices=["balanced", "proportional"],
        help="When --total_samples is set: balanced (default) or proportional allocation across hops.",
    )
    p_prepare.add_argument("--sample_seed", type=int, default=13, help="Seed for per-hop sampling.")

    p_matrix = subparsers.add_parser("matrix", help="Generate and optionally execute method x hop x seed runs.")
    p_matrix.add_argument("--data_manifest", required=True, help="Path to prepare-step manifest.json.")
    p_matrix.add_argument("--run_root", required=True, help="Run root directory.")
    p_matrix.add_argument("--repo_root", default=str(REPO_ROOT), help="Repository root.")
    p_matrix.add_argument("--config", default="", help="Base config path (default: relrag/config/config.yaml).")
    p_matrix.add_argument("--methods", default="standard,relrag", help="Methods list: standard,relrag,relrag_no_walk")
    p_matrix.add_argument("--hops", default="2,3,4", help="Hop list.")
    p_matrix.add_argument("--seeds", default="11,29,47", help="Seed list.")
    p_matrix.add_argument("--reader", default="vllm", choices=["vllm", "openai"], help="Reader backend.")
    p_matrix.add_argument("--standard_retriever", default="dense", help="standard retriever: bm25|dense")
    p_matrix.add_argument("--relrag_retriever", default="hybrid", help="relrag retriever: bm25|dense|hybrid|structured")
    p_matrix.add_argument("--split", default="test", help="Split label for run metadata.")
    p_matrix.add_argument("--workers", type=int, default=1, help="Workers per run.")
    p_matrix.add_argument("--top_k", type=int, default=10, help="Final evidence top-k.")
    p_matrix.add_argument("--top_k_raw", type=int, default=20, help="Raw retrieval top-k before dedup.")
    p_matrix.add_argument("--min_overfetch", type=float, default=2.0, help="Minimum overfetch factor.")
    p_matrix.add_argument("--backfill_max_overfetch", type=float, default=4.0, help="Backfill max overfetch.")
    p_matrix.add_argument("--backfill_step", type=float, default=1.5, help="Backfill growth step.")
    p_matrix.add_argument("--backfill_rounds", type=int, default=3, help="Backfill rounds.")
    p_matrix.add_argument("--llm_retry_on_empty", type=int, default=1, help="LLM retry count.")
    p_matrix.add_argument("--llm_retry_max_evidence", type=int, default=6, help="Max evidences used in retry.")
    p_matrix.add_argument("--include_decomposition_sp", action="store_true", help="Use decomposition support idx in gold_sp.")
    p_matrix.add_argument("--no_decomposition_sp", action="store_false", dest="include_decomposition_sp", help="Disable decomposition support idx in gold_sp.")
    p_matrix.set_defaults(include_decomposition_sp=True)
    p_matrix.add_argument("--endpoint", default="", help="Override LLM endpoint.")
    p_matrix.add_argument("--model", default="", help="Override LLM model id.")
    p_matrix.add_argument("--emb_endpoint", default="", help="Override embedding endpoint via EMB_ENDPOINT.")
    p_matrix.add_argument("--openai_model", default="", help="OpenAI model override.")
    p_matrix.add_argument("--openai_temperature", type=float, default=None, help="OpenAI temperature override.")
    p_matrix.add_argument("--openai_max_tokens", type=int, default=None, help="OpenAI max tokens override.")
    p_matrix.add_argument("--limit", type=int, default=0, help="Optional per-run limit.")
    p_matrix.add_argument("--dry_run", action="store_true", help="Write matrix and commands, do not execute.")
    p_matrix.add_argument("--resume", action="store_true", help="Skip run if completed.json exists.")
    p_matrix.add_argument("--heartbeat_sec", type=float, default=20.0, help="Print running heartbeat every N seconds (0 to disable).")
    p_matrix.add_argument("--progress", action="store_true", help="Print per-run progress logs to terminal.")
    p_matrix.add_argument("--no_progress", action="store_false", dest="progress", help="Disable terminal progress logs.")
    p_matrix.set_defaults(progress=True)

    p_analyze = subparsers.add_parser("analyze", help="Aggregate hop-stratified results and draw curves.")
    p_analyze.add_argument("--run_root", required=True, help="Run root directory.")
    p_analyze.add_argument("--output_dir", required=True, help="Output directory for analysis artifacts.")
    p_analyze.add_argument("--bootstrap_samples", type=int, default=5000, help="Bootstrap samples.")
    p_analyze.add_argument("--bootstrap_seed", type=int, default=2026, help="Bootstrap seed.")
    p_analyze.add_argument("--ci", type=float, default=0.95, help="Confidence interval level.")

    args = parser.parse_args()
    if args.cmd == "prepare":
        _prepare_dataset(args)
    elif args.cmd == "matrix":
        _build_matrix(args)
    elif args.cmd == "analyze":
        _analyze_runs(args)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported command {args.cmd}")


if __name__ == "__main__":
    main()
