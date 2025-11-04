from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Seed:
    text: str
    type_hint: Optional[str] = None


@dataclass
class PredicateStep:
    pred: str
    direction: str = "out"  # "out" or "in"
    target_hint: Optional[str] = None


@dataclass
class QueryIR:
    intent: str
    seeds: List[Seed]
    pred_chain: List[PredicateStep] = field(default_factory=list)
    target_type: Optional[str] = None
    question_type: Optional[str] = None
    max_hops: int = 2
    fanout: int = 10
    raw: str = ""
    fallback: bool = False

    @property
    def is_valid(self) -> bool:
        return bool(self.seeds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "seeds": [seed.__dict__ for seed in self.seeds],
            "pred_chain": [step.__dict__ for step in self.pred_chain],
            "target_type": self.target_type,
            "question_type": self.question_type,
            "max_hops": self.max_hops,
            "fanout": self.fanout,
            "fallback": self.fallback,
            "raw": self.raw,
        }
