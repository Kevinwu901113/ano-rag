
import os

file_path = "/home/wjk/workplace/nq/ano-rag/IE_Comparison_Report.md"
output_lines = []

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    stripped = line.strip()
    
    if stripped.startswith("| K |"):
        # Header
        output_lines.append("| K | Exp15 Dense | Exp15 BM25 | Exp15 Hybrid | Baseline Dense | Baseline BM25 | Structuredrag |\n")
        
    elif stripped.startswith("|---|"):
        # Separator
        output_lines.append("|---|---|---|---|---|---|---|\n")
        
    elif stripped.startswith("|") and len(stripped) > 2 and stripped[1:].strip()[0].isdigit():
        # Data row
        # Split by |
        parts = [p.strip() for p in line.split("|") if p.strip()]
        # parts[0] is K, parts[1] is Dense, parts[2] is BM25
        if len(parts) >= 3:
            try:
                k = parts[0]
                dense = float(parts[1])
                bm25 = float(parts[2])
                hybrid = parts[3]
                base_dense = parts[4]
                base_bm25 = parts[5]
                
                avg = (dense + bm25) / 2
                new_line = f"| {k} | {dense:.4f} | {bm25:.4f} | {hybrid} | {base_dense} | {base_bm25} | {avg:.4f} |\n"
                output_lines.append(new_line)
            except (ValueError, IndexError):
                output_lines.append(line)
        else:
            output_lines.append(line)
            
    else:
        output_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print(f"Updated {file_path}")
