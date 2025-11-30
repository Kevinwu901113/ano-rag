import re
from baselines.naive_rag.runner import _enforce_short_answer, _strip_reasoning

def test_cleaning():
    test_cases = [
        ("The answer is Paris.", "Paris"),
        ("Based on the context, the answer is Paris.", "Paris"),
        ("Occupation is actress.", "actress"),
        ("Insufficient evidence", "Insufficient evidence"),
        ("I cannot answer this based on the provided context.", "I cannot answer this based on the provided context."), # Should NOT become Insufficient evidence automatically unless caught
        ("Answer: Paris", "Paris"),
        ("Reasoning: blah blah\nAnswer: Paris", "Paris"),
        ("<think>some thought</think>Paris", "Paris"),
        ("The user is asking for the capital. The answer is Paris.", "Paris"),
    ]

    print("Testing _enforce_short_answer and _strip_reasoning...")
    for input_text, expected in test_cases:
        # Simulate the pipeline
        stripped = _strip_reasoning(input_text)
        result = _enforce_short_answer(stripped)
        print(f"Input: {input_text!r}")
        print(f"Output: {result!r}")
        print(f"Expected: {expected!r}")
        if result != expected:
            print("MATCH FAILED")
        else:
            print("MATCH OK")
        print("-" * 20)

if __name__ == "__main__":
    test_cleaning()
