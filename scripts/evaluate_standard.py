import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Set
from loguru import logger

# Add root to sys.path
ROOT = Path(__file__).resolve().parents[1] # scripts/ -> root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import our new standard metrics
from utils.metrics.retrieval import compute_retrieval_metrics, aggregate_retrieval_metrics
from utils.metrics.generation import compute_generation_metrics
# from utils.metrics.quality import RAGQualityEvaluator # Optional, expensive

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    if not path.exists():
        return data
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_gold_dataset(dataset_path: Path, dataset_type: str) -> Dict[str, Any]:
    """
    Load gold data and normalize canonical IDs.
    Returns:
        qa_map: {qid: [answers]}
        retrieval_map: {qid: {ids}} (Set of canonical IDs)
        canonical_type: "doc_id" or "passage_id"
    """
    data = []
    if str(dataset_path).endswith('.jsonl'):
        data = load_jsonl(dataset_path)
    else:
        with dataset_path.open('r') as f:
            data = json.load(f)
            
    qa_map = {}
    retrieval_map = {}
    canonical_type = "doc_id" # Default
    
    if dataset_type == "hotpotqa":
        canonical_type = "doc_id" # Title is Doc ID
        for item in data:
            qid = str(item.get("id") or item.get("_id"))
            ans = item.get("answer")
            qa_map[qid] = [str(ans)] if ans else []
            
            # Retrieval Gold: Supporting Facts titles
            supp = item.get("supporting_facts") or []
            # supp is list of [title, sent_id] or dict
            if isinstance(supp, list):
                # Unique titles
                titles = set(x[0] for x in supp)
                retrieval_map[qid] = titles
            elif isinstance(supp, dict):
                 retrieval_map[qid] = set(supp.get("title", []))
                 
    elif dataset_type == "musique":
        canonical_type = "passage_id" # PID
        for item in data:
            qid = str(item.get("id"))
            # Answers
            ans_list = []
            if item.get("answer"): ans_list.append(item["answer"])
            ans_list.extend(item.get("answer_aliases", []))
            qa_map[qid] = ans_list
            
            # Retrieval Gold: Paragraph PIDs
            pids = set()
            for idx, p in enumerate(item.get("paragraphs", [])):
                if p.get("is_supporting"):
                    pid = p.get("pid") or p.get("id") or f"p{idx:04d}"
                    pids.add(str(pid))
            retrieval_map[qid] = pids
            
    elif dataset_type == "mirage":
        canonical_type = "doc_id" # UUID
        # Load doc_pool for mapping if needed? No, prompt says 'mapped_id' is in dataset?
        # Actually _load_mirage in previous script used doc_name or doc_id from dataset.
        for item in data:
            qid = str(item.get("query_id") or item.get("id"))
            ans = item.get("answer")
            qa_map[qid] = [str(ans)] if isinstance(ans, str) else (ans or [])
            
            # Retrieval Gold
            # For Mirage, query_id IS the mapped_id (Canonical ID)
            # We ignore doc_name for retrieval matching because artifacts use mapped_id (UUID)
            if qid:
                retrieval_map[qid] = {qid}
            else:
                retrieval_map[qid] = set()
                
    else:
        logger.warning(f"Unknown dataset type: {dataset_type}")
        
    return qa_map, retrieval_map, canonical_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--dataset-type", required=True, choices=["hotpotqa", "musique", "mirage"])
    parser.add_argument("--run-rag-quality", action="store_true", help="Run expensive RAG quality evaluation")
    args = parser.parse_args()
    
    run_dir = args.run_dir
    if not run_dir.exists():
        logger.error(f"Run dir not found: {run_dir}")
        return

    # Load Gold
    qa_gold, ret_gold, canonical_type = load_gold_dataset(args.dataset_path, args.dataset_type)
    logger.info(f"Loaded gold for {len(qa_gold)} queries. Canonical ID Type: {canonical_type}")
    
    # --- Retrieval Evaluation ---
    retrieval_file = run_dir / "retrieval.jsonl"
    # If not at root, try artifacts
    if not retrieval_file.exists():
        retrieval_file = run_dir / "artifacts" / "retrieval.jsonl"
        
    ret_metrics_all = []
    
    if retrieval_file.exists():
        logger.info(f"Evaluating retrieval from {retrieval_file}")
        records = load_jsonl(retrieval_file)
        
        for rec in records:
            qid = str(rec.get("id"))
            if qid not in ret_gold:
                continue
                
            retrieved = rec.get("retrieved", [])
            # IMPORTANT: For Mirage, we need to strip 'mirage/' prefix if present in doc_id
            # to match the dataset gold which usually doesn't have it (or check consistency).
            # The audit showed Mirage fails because of this.
            # We apply a fix here: if dataset_type is mirage, strip 'mirage/' from doc_id.
            if args.dataset_type == "mirage":
                for item in retrieved:
                    did = item.get("doc_id", "")
                    if str(did).startswith("mirage/"):
                        # Extract the UUID part. mirage/UUID::Index or mirage/UUID
                        clean = str(did).replace("mirage/", "")
                        if "::" in clean:
                            clean = clean.split("::")[0]
                        item["doc_id"] = clean
            
            if args.dataset_type == "mirage" and len(ret_metrics_all) < 3:
                 logger.info(f"Mirage Debug - QID: {qid}")
                 logger.info(f"  Gold IDs: {ret_gold[qid]}")
                 logger.info(f"  Retrieved IDs (Cleaned): {[x.get('doc_id') for x in retrieved[:3]]}")
            
            # Compute
            # Note: For Musique, canonical_type is passage_id. We must ensure retrieved items have passage_id.
            # If retrieved items don't have passage_id, we might fallback to doc_id or fail.
            m = compute_retrieval_metrics(
                retrieved, 
                ret_gold[qid], 
                k_list=[1, 3, 5, 10], 
                id_key=canonical_type
            )
            ret_metrics_all.append(m)
            
        # Aggregate
        agg_ret = aggregate_retrieval_metrics(ret_metrics_all)
        agg_ret["count"] = len(ret_metrics_all)
        
        # Save
        save_json(agg_ret, run_dir / "metrics" / "retrieval_metrics.json")
        logger.info("Saved retrieval metrics.")
    else:
        logger.warning("No retrieval.jsonl found.")

    # --- Generation Evaluation ---
    # Try to find predictions
    pred_files = [
        run_dir / "preds" / "pred.json",            # Hotpot
        run_dir / "preds" / "answers.json",         # Mirage old
        run_dir / "preds" / "musique_results.jsonl", # Musique
        run_dir / "preds" / "results.jsonl",        # Mirage new
    ]
    
    preds_map = {}
    
    for pfile in pred_files:
        if pfile.exists():
            logger.info(f"Loading predictions from {pfile}")
            try:
                if pfile.suffix == '.jsonl':
                    records = load_jsonl(pfile)
                    for item in records:
                        qid = item.get("query_id") or item.get("id")
                        ans = item.get("answer") or item.get("predicted_answer") or item.get("pred_raw")
                        if qid and ans:
                            preds_map[str(qid)] = ans
                else:
                    with pfile.open('r') as f:
                        pdata = json.load(f)
                        if isinstance(pdata, dict):
                            if "answer" in pdata:
                                preds_map = pdata["answer"]
                            else:
                                preds_map = pdata
                        elif isinstance(pdata, list):
                            for item in pdata:
                                if isinstance(item, dict):
                                    qid = item.get("query_id") or item.get("id")
                                    ans = item.get("answer") or item.get("predicted_answer")
                                    if qid and ans:
                                        preds_map[str(qid)] = ans
                if preds_map:
                    break # Stop if we found predictions
            except Exception as e:
                logger.warning(f"Failed to load predictions from {pfile}: {e}")

    # We also look for qa.tsv as fallback or primary
    qa_file = run_dir / "preds" / "qa.tsv"
    
    
    gen_metrics_all = []
    
    if preds_map:
        logger.info(f"Evaluating generation for {len(preds_map)} predictions")
        em_list = []
        f1_list = []
        
        for qid, pred_ans in preds_map.items():
            if qid not in qa_gold:
                continue
            
            m = compute_generation_metrics(str(pred_ans), qa_gold[qid])
            em_list.append(m["EM"])
            f1_list.append(m["F1"])
            
        if em_list:
            agg_gen = {
                "EM": sum(em_list) / len(em_list) * 100.0,
                "F1": sum(f1_list) / len(f1_list) * 100.0,
                "count": len(em_list)
            }
            save_json(agg_gen, run_dir / "metrics" / "generation_metrics.json")
            logger.info("Saved generation metrics.")
            
            # Format Metrics
            invalid_count = sum(1 for a in preds_map.values() if not a or str(a).lower() == "error")
            format_metrics = {
                "invalid_rate": invalid_count / len(preds_map) if preds_map else 0.0,
                "no_final_tag_rate": 0.0, # Placeholder as we don't check tags yet
                "total_predictions": len(preds_map)
            }
            save_json(format_metrics, run_dir / "metrics" / "format_metrics.json")
    else:
        logger.warning("No predictions found.")

    # --- RAG Quality (Optional) ---
    if args.run_rag_quality:
        logger.info("Running RAG Quality Evaluation (Mock/Dry run if no LLM config)")
        # This requires reading context from pred_raw.jsonl or similar
        pass

if __name__ == "__main__":
    main()
