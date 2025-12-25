from __future__ import annotations


FINAL_TAG = "FINAL:"


def build_final_instruction() -> str:
    return (
        "Your output MUST end with exactly one line that starts with FINAL: followed by the answer. "
        "Do NOT put the final answer inside <think>."
    )
