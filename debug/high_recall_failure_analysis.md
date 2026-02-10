# MuSiQue High Recall Low F1 Failure Analysis

## Summary
We analyzed specific cases where Retrieval Recall was perfect (1.0) but QA F1 was 0.0. These cases indicate "Reader Bottlenecks" where the LLM fails to utilize the retrieved information correctly.

## Case 1: Ambiguity & World Knowledge Conflict
**ID**: `2hop__66167_615257`
**Question**: "What team does the player with the most points in a NBA season play for?"
**Gold Answer**: "Oklahoma City Thunder" (Implies Kevin Durant, 2009-10 season)
**Prediction**: "Philadelphia Warriors" (Implies Wilt Chamberlain, 1961-62 season)

**Analysis**:
- **Retrieved Context**: Contains BOTH:
    - [Gold] Kevin Durant: "youngest scoring leader... in the 2009-10 season".
    - [Distractor/True] Wilt Chamberlain: "holds the all-time records for total points... in a season... 1961-62".
- **Failure Reason**: The question asks for "most points in **a** NBA season" without specifying "2009-10" or "current".
- **Interpretation**: The model answered the *literal, all-time* factual question correctly (Wilt Chamberlain -> Philadelphia Warriors). The Gold Answer relies on an implicit constraint (likely "in the context of the 2009-10 season") which was not present in the question.
- **Verdict**: **Valid Model Behavior / Dataset Ambiguity**. The model prioritized the "All-time" superlative over the "Context-specific" one.

## Case 2: Semantic Mismatch & Reasoning Strictness
**ID**: `4hop2__103790_14670_8987_8529`
**Question**: "...military branch... was unprepared for the invasion of Ivo Werner's country [Czechoslovakia]. When was the word 'Slavs' used in the national anthem of the unprepared country?"
**Gold Answer**: "1943–1992" (Yugoslavia)
**Prediction**: "Insufficient evidence"

**Analysis**:
- **Reasoning Chain**:
    1. Ivo Werner -> Czechoslovakia.
    2. Invasion of Czechoslovakia -> Unpreparedness of Yugoslav army.
    3. Unprepared Country -> Yugoslavia.
    4. Anthem of Yugoslavia -> 1943-1992.
- **Retrieved Context**: "Tito removed generals... in the aftermath of the invasion of Czechoslovakia due to the **unpreparedness of the Yugoslav army to respond to a similar invasion of Yugoslavia**."
- **Failure Reason**:
    - The Question claims the branch was "unprepared **for the invasion of Czechoslovakia**".
    - The Text states the branch was "unprepared **to respond to a similar invasion of Yugoslavia**".
- **Interpretation**: The model likely detected this semantic mismatch. The Yugoslav army wasn't necessarily unprepared *for the Czech invasion* (it might not have been involved), but was unprepared for its *own* defense.
- **Verdict**: **Reasoning Strictness**. The model refused to answer because the premise in the question ("unprepared for invasion of Czechoslovakia") was not strictly supported by the text ("unprepared for invasion of Yugoslavia"), leading to "Insufficient evidence".

## Conclusion
High Recall failures in MuSiQue are often due to:
1.  **Ambiguous Superlatives**: Questions using "most/best/first" without scope constraints (All-time vs Context).
2.  **Semantic Precision**: The model (especially strong ones like Qwen/DeepSeek) being pedantic about the exact relationship between entities, rejecting loose phrasing in generated questions.

## Recommendations
- **Prompt Engineering**: Instruct the model to "Prioritize information found in the context over general world knowledge" (to fix Case 1).
- **Prompt Engineering**: Instruct the model to "Allow for reasonable semantic inference even if phrasing is slightly inexact" (to fix Case 2).
- **Evaluation**: Accept that some F1=0 cases are actually valid refusals or valid alternative interpretations.
