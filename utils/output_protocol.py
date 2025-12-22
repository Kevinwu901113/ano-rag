from __future__ import annotations


FINAL_TAG = "FINAL:"


def build_final_instruction() -> str:
    return (
        "Respond with exactly one line in the form: FINAL: <answer>. "
        "Do not include any other text. Otherwise the result is invalid."
    )
