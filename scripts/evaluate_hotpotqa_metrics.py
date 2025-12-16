import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


def clean_prediction(text: str) -> str:
    """
    Clean the prediction text by removing common prefixes/suffixes 
    and extracting the actual answer from verbose outputs.
    """
    if not text:
        return ""
    text = str(text)
    
    # Handle "Answer: ..." (case insensitive)
    # We take the last part after "Answer:" as the model might output "Question: ... Answer: ..."
    if "answer:" in text.lower():
        parts = re.split(r"answer\s*:", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            candidate = parts[-1].strip()
            if candidate:
                text = candidate
    
    # Handle "The answer is ..."
    if "the answer is" in text.lower():
        parts = re.split(r"the answer is\s*", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            candidate = parts[-1].strip()
            if candidate:
                text = candidate

    return text.strip()


def normalize(text: str) -> str:
    """Standard HotpotQA/SQuAD normalization."""
    def remove_articles(t):
        return re.sub(r'\b(a|an|the)\b', ' ', t)

    def white_space_fix(t):
        return ' '.join(t.split())

    def remove_punc(t):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in t if ch not in exclude)

    def lower(t):
        return t.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_prec_recall(pred: str, gold: str) -> Sequence[float]:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


def load_ground_truth(dataset_path: Path) -> Dict[str, Dict[str, object]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    answers = {str(item["id"]): str(item.get("answer", "")) for item in data}
    titles: Dict[str, Set[str]] = {}
    for item in data:
        sid = str(item["id"])
        supp = item.get("supporting_facts") or {}
        gold_titles = set(supp.get("title") or [])
        titles[sid] = gold_titles
    return {"answers": answers, "titles": titles}


def iter_predictions(pred_obj) -> Iterable[tuple]:
    if isinstance(pred_obj, list):
        for item in pred_obj:
            if not isinstance(item, dict):
                continue
            pid = item.get("id") or item.get("_id")
            if pid is None:
                continue
            yield str(pid), item.get("pred") or item.get("answer") or ""
    elif isinstance(pred_obj, dict):
        if "answer" in pred_obj and isinstance(pred_obj["answer"], dict):
            for pid, pred in pred_obj["answer"].items():
                yield str(pid), pred
        else:
            all_str = all(isinstance(v, str) for v in pred_obj.values())
            if all_str:
                for pid, pred in pred_obj.items():
                    yield str(pid), pred


def evaluate_predictions(pred_path: Path, answers: Dict[str, str]) -> Dict[str, float]:
    with pred_path.open("r", encoding="utf-8") as f:
        preds = json.load(f)
    em = 0
    sum_p = sum_r = sum_f1 = 0.0
    n = 0
    for pid, pred in iter_predictions(preds):
        gold = answers.get(pid)
        if gold is None:
            continue
        n += 1
        
        # Clean prediction
        pred = clean_prediction(pred)
        
        if normalize(str(pred)) == normalize(str(gold)):
            em += 1
        p, r, f1 = f1_prec_recall(str(pred), str(gold))
        sum_p += p
        sum_r += r
        sum_f1 += f1
    if n == 0:
        return {"EM": 0.0, "AnswerP": 0.0, "AnswerR": 0.0, "AnswerF1": 0.0, "count": 0}
    return {
        "EM": em / n,
        "AnswerP": sum_p / n,
        "AnswerR": sum_r / n,
        "AnswerF1": sum_f1 / n,
        "count": n,
    }


def sort_retrieved(items: List[dict]) -> List[dict]:
    def score(x: dict) -> float:
        s = x.get("score")
        if isinstance(s, (int, float)):
            return float(s)
        return float("-inf")

    return sorted(items, key=score, reverse=True)


def topk_titles(retrieved: List[dict], k: int) -> Set[str]:
    sorted_items = sort_retrieved(retrieved)
    titles = [str(it.get("title", "")).strip() for it in sorted_items[:k]]
    return {t for t in titles if t}


def _norm_for_hit(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    # Punctuation to space
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    return " ".join(text.split())


def _answer_pattern(ans_norm: str) -> re.Pattern:
    # Word boundary strict match
    return re.compile(rf"\b{re.escape(ans_norm)}\b", flags=re.IGNORECASE)


def _extract_hit_text(hit: dict) -> str:
    return str(hit.get("text") or hit.get("content") or hit.get("chunk") or "")


def answer_hit_at_k(hits: list, gold_answer: str, k: int) -> int:
    if not gold_answer:
        return 0
    
    gold_norm = _norm_for_hit(gold_answer)
    if len(gold_norm) < 4:  # Strict length check
        return 0
        
    pattern = _answer_pattern(gold_norm)
    
    # Ensure hits are sorted (they are sorted in loop usually, but let's be safe or rely on caller)
    # Actually evaluate_retrieval calls topk_titles which sorts. 
    # Here we should assume hits passed to us might not be sorted if we pass raw list?
    # Better to sort here or assume sorted. 
    # In evaluate_retrieval loop, we should sort once.
    
    # For efficiency, let's assume the caller passes sorted hits or we sort.
    # To be safe and match logic:
    sorted_hits = sort_retrieved(hits)
    
    for hit in sorted_hits[:k]:
        text = _norm_for_hit(_extract_hit_text(hit))
        if pattern.search(text):
            return 1
    return 0


def evaluate_retrieval(
    retrieval_path: Path, 
    gold_title_sets: Dict[str, Set[str]], 
    gold_answers: Dict[str, str],
    ks: Iterable[int] = (5, 10)
) -> Dict[str, float]:
    metrics = {f"TitleRecall@{k}": 0.0 for k in ks}
    metrics.update({f"TitlePrec@{k}": 0.0 for k in ks})
    metrics.update({f"Hit@{k}": 0.0 for k in ks})
    metrics.update({f"AnswerHit@{k}": 0.0 for k in ks})
    n = 0
    n_ans = 0 # Count queries with valid answers for AnswerHit

    with retrieval_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = str(obj.get("id"))
            gold_titles = gold_title_sets.get(pid)
            if gold_titles is None:
                continue
            
            retrieved = obj.get("retrieved") or []
            final_context = obj.get("final_context") or []
            
            if final_context:
                # Map text from final_context to retrieved items
                text_map = {}
                for fc in final_context:
                    # Normalized logger puts title in 'title' or 'doc_id'
                    t = str(fc.get("title") or fc.get("doc_id") or "").strip()
                    if t:
                        text_map[t] = fc.get("text", "")
                
                for hit in retrieved:
                    t = str(hit.get("title") or hit.get("doc_id") or "").strip()
                    if t and t in text_map:
                        hit["text"] = text_map[t]

            # Sort once for consistency
            retrieved = sort_retrieved(retrieved)
            
            if idx == 0 and retrieved:
                print(f"DEBUG [{retrieval_path.parent.name}] hit keys:", retrieved[0].keys())
                print(f"DEBUG [{retrieval_path.parent.name}] text preview:", _extract_hit_text(retrieved[0])[:200])

            n += 1
            
            # Title Metrics
            for k in ks:
                # Retrieved is already sorted
                top_k_items = retrieved[:k]
                retrieved_titles = {str(it.get("title", "")).strip() for it in top_k_items if it.get("title")}
                
                inter = len(retrieved_titles & gold_titles)
                if gold_titles:
                    recall = inter / len(gold_titles)
                else:
                    recall = 1.0 if not retrieved_titles else 0.0
                if retrieved_titles:
                    prec = inter / len(retrieved_titles)
                else:
                    prec = 1.0 if not gold_titles else 0.0
                hit = 1.0 if inter > 0 else (1.0 if not gold_titles and not retrieved_titles else 0.0)
                metrics[f"TitleRecall@{k}"] += recall
                metrics[f"TitlePrec@{k}"] += prec
                metrics[f"Hit@{k}"] += hit

            # AnswerHit Metrics
            gold_ans = gold_answers.get(pid)
            if gold_ans:
                n_ans += 1
                for k in ks:
                    # We pass retrieved (sorted) but answer_hit_at_k sorts again? 
                    # Let's optimize: answer_hit_at_k sorts. It's fine.
                    if answer_hit_at_k(retrieved, gold_ans, k):
                        metrics[f"AnswerHit@{k}"] += 1

    if n == 0:
        return {}
        
    out = {k: v / n for k, v in metrics.items() if not k.startswith("AnswerHit")}
    if n_ans > 0:
        out.update({k: v / n_ans for k, v in metrics.items() if k.startswith("AnswerHit")})
    else:
        out.update({k: 0.0 for k, v in metrics.items() if k.startswith("AnswerHit")})
        
    return out


def render_answer_table(metrics: Dict[str, Dict[str, float]]) -> str:
    header = "| run | EM | AnswerP | AnswerR | AnswerF1 | count |\n"
    header += "| --- | --- | --- | --- | --- | --- |\n"
    rows = []
    for run, m in sorted(metrics.items()):
        rows.append(
            f"| {run} | {m.get('EM', 0):.3f} | {m.get('AnswerP', 0):.3f} | "
            f"{m.get('AnswerR', 0):.3f} | {m.get('AnswerF1', 0):.3f} | {int(m.get('count', 0))} |"
        )
    return header + "\n".join(rows)


def render_retrieval_table(metrics: Dict[str, Dict[str, float]]) -> str:
    header = (
        "| run | TitleRecall@5 | TitleRecall@10 | Hit@5 | Hit@10 | AnswerHit@5 | AnswerHit@10 |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = []
    for run, m in sorted(metrics.items()):
        has_retrieval = any(key.startswith("TitleRecall@") for key in m)
        if not has_retrieval:
            continue
        rows.append(
            f"| {run} | {m.get('TitleRecall@5', 0):.3f} | {m.get('TitleRecall@10', 0):.3f} | "
            f"{m.get('Hit@5', 0):.3f} | {m.get('Hit@10', 0):.3f} | "
            f"{m.get('AnswerHit@5', 0):.3f} | {m.get('AnswerHit@10', 0):.3f} |"
        )
    if not rows:
        return ""
    return header + "\n" + "\n".join(rows)


def collect_runs(root: Path) -> List[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("hotpot_"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA predictions and retrieval logs.")
    parser.add_argument(
        "--dataset",
        default="data/hotpotqa/dataset_distractor_200.json",
        help="Ground truth dataset with answers and supporting_facts.",
    )
    parser.add_argument(
        "--root",
        default="result/hotpotqa",
        help="Root directory containing hotpot_* subdirectories.",
    )
    parser.add_argument(
        "--output",
        default="hotpot_metrics.json",
        help="Path to write aggregated metrics JSON.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    root = Path(args.root)
    output_path = Path(args.output)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    gt = load_ground_truth(dataset_path)
    metrics: Dict[str, Dict[str, float]] = {}

    for run_dir in collect_runs(root):
        pred_path = run_dir / "pred.json"
        if not pred_path.exists():
            continue
        run_metrics = evaluate_predictions(pred_path, gt["answers"])
        retrieval_path = run_dir / "retrieval.jsonl"
        if retrieval_path.exists():
            retrieval_metrics = evaluate_retrieval(retrieval_path, gt["titles"], gt["answers"])
            run_metrics.update(retrieval_metrics)
        metrics[run_dir.name] = run_metrics

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    answer_table = render_answer_table(metrics)
    print("Answer metrics:")
    print(answer_table)

    retrieval_table = render_retrieval_table(metrics)
    if retrieval_table:
        print("\nRetrieval metrics:")
        print(retrieval_table)


if __name__ == "__main__":
    main()
