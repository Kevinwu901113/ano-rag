import json
import os
import re
import string
import sys
from collections import Counter
from statistics import median
from typing import Any, Dict, Iterable, List, Tuple

# Add scripts directory to path for calc_musique_recall
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))
from calc_musique_recall import calculate_recall


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    zero_metric = (0.0, 0.0, 0.0)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def update_answer(metrics: Dict[str, float], prediction: str, gold: str) -> Tuple[float, float, float]:
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics["em"] += float(em)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall
    return float(em), prec, recall


def update_sp(metrics: Dict[str, float], prediction: List[List[Any]], gold: List[List[Any]]) -> Tuple[float, float, float]:
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp = sum(1 for e in cur_sp_pred if e in gold_sp_pred)
    fp = sum(1 for e in cur_sp_pred if e not in gold_sp_pred)
    fn = sum(1 for e in gold_sp_pred if e not in cur_sp_pred)
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics["sp_em"] += em
    metrics["sp_f1"] += f1
    metrics["sp_prec"] += prec
    metrics["sp_recall"] += recall
    return em, prec, recall


def eval_hotpot(prediction: Dict[str, Dict[str, Any]], gold: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics = {
        "em": 0.0,
        "f1": 0.0,
        "prec": 0.0,
        "recall": 0.0,
        "sp_em": 0.0,
        "sp_f1": 0.0,
        "sp_prec": 0.0,
        "sp_recall": 0.0,
        "joint_em": 0.0,
        "joint_f1": 0.0,
        "joint_prec": 0.0,
        "joint_recall": 0.0,
    }

    pred_ids = set(prediction["answer"].keys())
    gold_subset = [dp for dp in gold if dp["_id"] in pred_ids]
    if not gold_subset:
        return metrics

    for dp in gold_subset:
        cur_id = dp["_id"]
        can_eval_joint = True

        em, prec, recall = update_answer(metrics, prediction["answer"][cur_id], dp["answer"])
        if cur_id not in prediction["sp"]:
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(metrics, prediction["sp"][cur_id], dp["supporting_facts"])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            joint_f1 = (
                2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                if joint_prec + joint_recall > 0
                else 0.0
            )
            joint_em = em * sp_em

            metrics["joint_em"] += joint_em
            metrics["joint_f1"] += joint_f1
            metrics["joint_prec"] += joint_prec
            metrics["joint_recall"] += joint_recall

    denom = float(len(gold_subset))
    for key in list(metrics.keys()):
        metrics[key] /= denom
    return metrics


def load_prediction_data(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    prediction = {"answer": {}, "sp": {}}
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                data = json.loads(line)
            except Exception:
                continue

            qid = data.get("_id") or data.get("id")
            if not qid:
                continue
            ans = str(data.get("prediction", ""))
            sp = data.get("pred_sp")
            if sp is None:
                sp = data.get("sp", [])

            cleaned_sp = []
            if isinstance(sp, list):
                for item in sp:
                    if isinstance(item, list) and len(item) >= 2:
                        cleaned_sp.append([item[0], item[1]])
            prediction["answer"][str(qid)] = ans
            prediction["sp"][str(qid)] = cleaned_sp
    return prediction


def _extract_titles(contexts: Any) -> List[str]:
    if not isinstance(contexts, list):
        return []
    titles = []
    for item in contexts:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("doc_title") or "").strip()
        if title:
            titles.append(title)
    return titles


def _first_rank(titles: List[str], target: str) -> int:
    for idx, title in enumerate(titles, start=1):
        if title == target:
            return idx
    return 10**9


def analyze_retrieval_bottleneck(pred_file: str) -> Dict[str, Any]:
    total = 0
    valid = 0
    second_in_raw = 0
    second_in_top5 = 0
    second_rank_raw: List[int] = []
    second_rank_topk: List[int] = []
    one_hit_top2 = 0
    one_hit_top5 = 0
    full_hit_top2 = 0
    full_hit_top5 = 0
    gold_size_sum = 0

    with open(pred_file, "r", encoding="utf-8") as handle:
        for line in handle:
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue

            gold_sp = row.get("gold_sp") or []
            gold_titles = sorted(
                {
                    str(item[0]).strip()
                    for item in gold_sp
                    if isinstance(item, list) and len(item) >= 1 and str(item[0]).strip()
                }
            )
            if not gold_titles:
                continue

            valid += 1
            gold_size_sum += len(gold_titles)
            raw_titles = _extract_titles(row.get("retrieved_context_raw"))
            topk_titles = _extract_titles(row.get("retrieved_context_topk"))
            gold_set = set(gold_titles)

            hit2 = sum(1 for title in gold_set if title in set(topk_titles[:2]))
            hit5 = sum(1 for title in gold_set if title in set(topk_titles[:5]))
            if hit2 >= 1:
                one_hit_top2 += 1
            if hit5 >= 1:
                one_hit_top5 += 1
            if hit2 == len(gold_set):
                full_hit_top2 += 1
            if hit5 == len(gold_set):
                full_hit_top5 += 1

            if len(gold_titles) >= 2:
                candidate = None
                for title in gold_titles:
                    if title not in set(topk_titles[:1]):
                        candidate = title
                        break
                if candidate is None:
                    candidate = gold_titles[0]

                rank_raw = _first_rank(raw_titles, candidate)
                rank_topk = _first_rank(topk_titles, candidate)
                if rank_raw < 10**9:
                    second_in_raw += 1
                    second_rank_raw.append(rank_raw)
                if rank_topk <= 5:
                    second_in_top5 += 1
                if rank_topk < 10**9:
                    second_rank_topk.append(rank_topk)

    if valid == 0:
        return {
            "count": 0,
            "avg_gold_titles": 0.0,
            "one_hit_top2_ratio": 0.0,
            "one_hit_top5_ratio": 0.0,
            "full_hit_top2_ratio": 0.0,
            "full_hit_top5_ratio": 0.0,
            "second_title_in_raw_ratio": 0.0,
            "second_title_in_top5_ratio": 0.0,
            "second_title_rank_raw_median": None,
            "second_title_rank_topk_median": None,
        }

    return {
        "count": valid,
        "avg_gold_titles": gold_size_sum / valid,
        "one_hit_top2_ratio": one_hit_top2 / valid,
        "one_hit_top5_ratio": one_hit_top5 / valid,
        "full_hit_top2_ratio": full_hit_top2 / valid,
        "full_hit_top5_ratio": full_hit_top5 / valid,
        "second_title_in_raw_ratio": second_in_raw / valid,
        "second_title_in_top5_ratio": second_in_top5 / valid,
        "second_title_rank_raw_median": median(second_rank_raw) if second_rank_raw else None,
        "second_title_rank_topk_median": median(second_rank_topk) if second_rank_topk else None,
    }


def main() -> None:
    base_dir = "/home/wjk/workplace/nq/ano-rag/result/experiment_21"
    gold_file = "/home/wjk/workplace/nq/ano-rag/data_full/hotpot_dev_distractor_v1.json"
    modes = ["bm25", "dense", "hybrid"]

    with open(gold_file, "r", encoding="utf-8") as handle:
        gold_data = json.load(handle)

    print("# Experiment 21 Evaluation Results\n")

    print("## 1. Core QA Metrics (VLLM Qwen3-30b)\n")
    print("| Retriever | EM | F1 | SP EM | SP F1 | Joint EM | Joint F1 |")
    print("|---|---|---|---|---|---|---|")

    recall_raw: Dict[str, Dict[str, float]] = {}
    recall_topk: Dict[str, Dict[str, float]] = {}
    bottlenecks: Dict[str, Dict[str, Any]] = {}

    for mode in modes:
        pred_file = os.path.join(base_dir, f"pred_dev_{mode}.jsonl")
        if not os.path.exists(pred_file):
            print(f"| {mode} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue

        pred_data = load_prediction_data(pred_file)
        metrics = eval_hotpot(pred_data, gold_data)
        print(
            f"| **{mode.capitalize()}** | {metrics['em']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['sp_em']:.4f} | {metrics['sp_f1']:.4f} | {metrics['joint_em']:.4f} | {metrics['joint_f1']:.4f} |"
        )

        raw_metrics, _ = calculate_recall(pred_file, context_field="retrieved_context_raw")
        topk_metrics, _ = calculate_recall(pred_file, context_field="retrieved_context_topk")
        if raw_metrics:
            recall_raw[mode] = raw_metrics
        if topk_metrics:
            recall_topk[mode] = topk_metrics
        bottlenecks[mode] = analyze_retrieval_bottleneck(pred_file)

    print("\n## 2. Retrieval Recall (Raw vs Final Top-k)\n")
    print("| Retriever | R@2(raw) | R@2(topk) | Delta | R@5(raw) | R@5(topk) | Delta |")
    print("|---|---|---|---|---|---|---|")
    for mode in modes:
        r_raw = recall_raw.get(mode, {})
        r_topk = recall_topk.get(mode, {})
        if not r_raw or not r_topk:
            continue
        r2_raw = r_raw.get("R@2", 0.0)
        r2_topk = r_topk.get("R@2", 0.0)
        r5_raw = r_raw.get("R@5", 0.0)
        r5_topk = r_topk.get("R@5", 0.0)
        print(
            f"| **{mode.capitalize()}** | {r2_raw:.4f} | {r2_topk:.4f} | {r2_topk - r2_raw:+.4f} | "
            f"{r5_raw:.4f} | {r5_topk:.4f} | {r5_topk - r5_raw:+.4f} |"
        )

    print("\n## 3. Bottleneck Diagnostics (final top-k focus)\n")
    print("| Retriever | Avg Gold Titles | >=1 hit@2 | >=1 hit@5 | full hit@2 | full hit@5 | 2nd title in raw | 2nd title in top5 | 2nd rank raw(median) | 2nd rank topk(median) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for mode in modes:
        stat = bottlenecks.get(mode)
        if not stat:
            continue
        raw_med = stat["second_title_rank_raw_median"]
        topk_med = stat["second_title_rank_topk_median"]
        raw_med_text = "N/A" if raw_med is None else f"{raw_med:.1f}"
        topk_med_text = "N/A" if topk_med is None else f"{topk_med:.1f}"
        print(
            f"| **{mode.capitalize()}** | {stat['avg_gold_titles']:.2f} | "
            f"{stat['one_hit_top2_ratio']:.4f} | {stat['one_hit_top5_ratio']:.4f} | "
            f"{stat['full_hit_top2_ratio']:.4f} | {stat['full_hit_top5_ratio']:.4f} | "
            f"{stat['second_title_in_raw_ratio']:.4f} | {stat['second_title_in_top5_ratio']:.4f} | "
            f"{raw_med_text} | {topk_med_text} |"
        )


if __name__ == "__main__":
    main()
