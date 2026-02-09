import json
import argparse
from relrag.eval.llm_judge_emf1 import parse_sample, EvalSample

line = '{"_id": "5a8e5f1f5542995a26add4d6", "question": "Which dog...", "answer": "Insufficient evidence", "prediction": "Insufficient evidence", "references": ["Sapsali"]}'
row = json.loads(line)

args = argparse.Namespace()
args.id_key = None
args.question_key = None
args.pred_key = "prediction"
args.gold_key = "references"

sample = parse_sample(row, args)
print(f"Pred: {sample.pred}")
print(f"Golds: {sample.golds}")
