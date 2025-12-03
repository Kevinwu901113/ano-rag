from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TreeNode:
    node_id: int
    level: int
    text: str
    children: List[int] = field(default_factory=list)
    is_leaf: bool = False
    descendant_chunk_ids: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "node_id": self.node_id,
            "level": self.level,
            "text": self.text,
            "children": self.children,
            "is_leaf": self.is_leaf,
            "descendant_chunk_ids": self.descendant_chunk_ids
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            node_id=data["node_id"],
            level=data["level"],
            text=data["text"],
            children=data.get("children", []),
            is_leaf=data.get("is_leaf", False),
            descendant_chunk_ids=data.get("descendant_chunk_ids", [])
        )
