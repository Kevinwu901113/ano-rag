from __future__ import annotations

from relrag.prompt import load_prompt


FINAL_TAG = "FINAL:"


def build_final_instruction() -> str:
    return load_prompt("final_instruction.txt")
