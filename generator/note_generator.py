from typing import Dict, List

import requests

from validators.note_validator import validate_and_normalize

PROMPT_TMPL = """You are an information extractor. From the following text block, extract up to 3 atomic facts as JSON objects with fields: subj, pred, obj, subj_type, obj_type, evidence, meta.
Rules: one relation per object; evidence must copy an exact sentence from the text; pred must be one of the allowed set or mapped from synonyms; types must be from {PERSON, WORK, ORG, PLACE, EVENT, CONCEPT, TIME}. If no reliable facts, return [].
Allowed predicate sets (synonyms in parentheses):
performed_by(recorded_by,artist), authored_by(written_by), spouse(married_to,partner), parent(father,mother), born_in(place_of_birth), located_in, member_of, acted_in(starring), produced_by, released_in, label, founded_by, headquartered_in, winner_of, part_of
Return ONLY a JSON array. No extra text.

TEXT:
<<<{chunk_text}>>>
DOC_CHUNK: "{doc_id}#{chunk_id}"
"""


class NoteGenerator:
    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call(self, prompt: str) -> str:
        response = requests.post(
            f"{self.endpoint}/chat/completions",
            json={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def generate_for_chunk(self, chunk: Dict) -> List[Dict]:
        doc_id, chunk_id = chunk["doc_id"], chunk["chunk_id"]
        prompt = PROMPT_TMPL.format(
            chunk_text=chunk["text"], doc_id=doc_id, chunk_id=chunk_id
        )
        raw = self._call(prompt)
        ok, notes, _ = validate_and_normalize(raw, doc_id, chunk_id)
        return notes if ok else []
