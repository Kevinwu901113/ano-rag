from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class AtomicNote:
    doc_id: str
    content: str
    metadata: Dict

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d
