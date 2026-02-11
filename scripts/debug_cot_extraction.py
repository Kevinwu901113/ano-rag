
import json
import re

def extract_final_answer(text):
    if not text:
        return ""
    text = str(text)
    # Check for FINAL: pattern
    match = re.search(r"FINAL:\s*(.*)$", text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return text.strip()

input_file = '/home/wjk/workplace/nq/ano-rag/result/musique_experiment_5/pred_dev_dense.jsonl'
output_file = '/home/wjk/workplace/nq/ano-rag/result/musique_experiment_5/debug_extraction.jsonl'

print(f"Debugging extraction for {input_file}...")

with open(input_file, 'r') as f, open(output_file, 'w') as out:
    for i, line in enumerate(f):
        if i >= 20: break # Check first 20
        data = json.loads(line)
        raw_llm = data.get('retrieved_context', '') # Wait, raw LLM output is usually in 'answer_source_detail' or similar if not directly available?
        # Actually in the pred file, 'generated_answer' might be the raw output if extraction failed?
        # Let's check 'answer_source_detail' -> 'source'
        
        # Looking at previous `head` output:
        # "answer_source_detail": {"llm_has_final": false, "source": "llm_raw:extract_final_answer", ...}
        # "generated_answer": "(...The question asks..."
        # "short_answer": "(...The question asks..."
        
        # It seems the extraction logic FAILED to find "FINAL:" and thus returned the whole text (or truncated part) as the answer.
        # But wait, did the model output "FINAL:"?
        
        # We need to see the FULL `generated_answer` or `llm_raw` if available.
        # The `pred_dev_dense.jsonl` has `generated_answer`.
        
        gen_ans = data.get('generated_answer', '')
        extracted = extract_final_answer(gen_ans)
        
        out.write(json.dumps({
            "id": data.get('_id'),
            "generated_answer_snippet": gen_ans[:100] + "...",
            "has_final_tag": "FINAL:" in gen_ans,
            "extracted": extracted
        }) + "\n")

print("Debug extraction complete.")
