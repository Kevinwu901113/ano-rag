# Strict Recall Evaluation Report

This report summarizes the evaluation of retrieval systems using the strict **Answer-in-Chunk Recall (AnswerHit)** metric. This metric verifies whether the retrieved chunks actually contain the gold answer string, serving as a diagnostic for potential hallucinations or retrieval mismatch.

## 1. HotpotQA (Distractor Setting)
**Dataset**: `data/hotpotqa/dataset_distractor_200.json` (200 samples)
**Method**: Vanilla RAG (Retrieval Only)

### Retrieval Metrics
| Run | TitleRecall@5 | TitleRecall@10 | Hit@5 | Hit@10 | AnswerHit@5 | AnswerHit@10 |
| --- | --- | --- | --- | --- | --- | --- |
| hotpot_test_strict_eval | 0.320 | 0.320 | 0.540 | 0.540 | 0.430 | 0.430 |

*   **Hit@10 (Title)**: 54.0% - The retriever found at least one supporting document title in top-10.
*   **AnswerHit@10**: 43.0% - The retriever found a chunk containing the exact answer string in top-10.
*   **Gap**: ~11%. This indicates that in ~20% of cases where we retrieved the right document title, the chunk text might not have explicitly contained the answer (or the answer was split/formatted differently).

## 2. Mirage (Sample)
**Dataset**: `data/mirage/dataset.json` (First 20 samples)
**Method**: Naive RAG (FAISS Index)

### Retrieval Metrics
| Run | Metric | @1 | @3 | @5 | @10 |
| --- | --- | --- | --- | --- | --- |
| mirage_naive_recall | DocHit | 0.800 | 0.950 | 1.000 | 1.000 |
| mirage_naive_recall | AnswerHit | 0.050 | 0.100 | 0.100 | 0.100 |

*   **DocHit@10**: 100% - The system is very good at finding the correct document by title.
*   **AnswerHit@10**: 10% - However, only 10% of the top-10 chunks actually contained the answer string.
*   **Diagnosis**: This massive gap suggests that while the document is relevant, the chunking strategy or the specific chunks retrieved do not cover the span containing the answer, or the answer format in the dataset doesn't match the text (e.g., date formats, aliases).

## Methodology
- **AnswerHit@k**: Calculates the percentage of questions where at least one of the top-k retrieved chunks contains the answer string (case-insensitive, strict word boundary).
- **DocHit@k / Hit@k**: Calculates the percentage of questions where at least one of the top-k retrieved items corresponds to a gold document title/ID.
