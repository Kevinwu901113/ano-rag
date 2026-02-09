import json
import re
from pathlib import Path

PROFILE_SCHEMA = {
    "type": "object",
    "required": ["type", "aliases"],
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                "PERSON",
                "WORK",
                "ORG",
                "PLACE",
                "EVENT",
                "CONCEPT",
                "TIME",
            ],
        },
        "aliases": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "default": [],
        },
        "nationality": {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
        },
        "birth": {"type": ["string", "null"], "minLength": 1},
        "death": {"type": ["string", "null"], "minLength": 1},
        "occupations": {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
        },
        "titles": {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
        },
        "categories": {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
        },
        "same_as": {
            "type": ["array", "null"],
            "items": {"type": "string", "minLength": 1},
        },
        "description": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

ATTRIBUTE_VALUE_SCHEMA = {
    "type": "object",
    "required": ["value"],
    "properties": {
        "value": {"type": "string", "minLength": 1},
        "normalized": {"type": ["string", "null"], "minLength": 1},
        "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "source": {"type": ["string", "null"], "minLength": 1},
        "evidence": {"type": ["string", "null"], "minLength": 4},
        "qualifiers": {"type": ["object", "null"]},
        "notes": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

ATTRIBUTE_SCHEMA = {
    "type": "object",
    "required": ["name", "values"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "values": {
            "type": "array",
            "minItems": 1,
            "items": ATTRIBUTE_VALUE_SCHEMA,
        },
        "role": {"type": ["string", "null"]},
        "target_type": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "issues": {"type": ["array", "null"], "items": {"type": "string"}},
        "has_definition": {"type": ["boolean", "null"]},
    },
    "additionalProperties": False,
}

NOTE_JSON_SCHEMA = {
    "type": "array",
    "minItems": 0,
    "items": {
        "type": "object",
        "required": [
            "subj",
            "pred",
            "obj",
            "subj_type",
            "obj_type",
            "evidence",
            "meta",
        ],
        "properties": {
            "note_id": {"type": ["string", "null"], "minLength": 1},
            "subj": {
                "type": "string",
                "minLength": 1,
                "not": {"pattern": "(?i)^(he|she|they|his|her|their)$"}
            },
            "pred": {"type": "string", "minLength": 1},
            "obj": {
                "type": "string",
                "minLength": 1,
                "not": {"pattern": "(?i)^(he|she|they|his|her|their)$"}
            },
            "subj_type": {
                "type": "string",
                "enum": [
                    "PERSON",
                    "WORK",
                    "ORG",
                    "PLACE",
                    "EVENT",
                    "CONCEPT",
                    "TIME",
                ],
            },
            "obj_type": {
                "type": "string",
                "enum": [
                    "PERSON",
                    "WORK",
                    "ORG",
                    "PLACE",
                    "EVENT",
                    "CONCEPT",
                    "TIME",
                ],
            },
            "evidence": {"type": "string", "minLength": 4},
            "meta": {
                "type": "object",
                "required": ["source", "confidence", "subject_profile"],
                "properties": {
                    "source": {"type": "string", "minLength": 1},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "subject_profile": PROFILE_SCHEMA,
                    "object_profile": {"anyOf": [PROFILE_SCHEMA, {"type": "null"}]},
                    "attribute": ATTRIBUTE_SCHEMA,
                    "quality": QUALITY_SCHEMA,
                    "quality_score": {
                        "type": ["number", "null"],
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "domain": {"type": ["string", "null"]},
                    "year": {"type": ["string", "number", "null"]},
                    "section_rank": {"type": ["integer", "null"], "minimum": 0},
                    "anchor": {"type": ["boolean", "null"]},
                    "validation": {"type": ["string", "null"], "minLength": 1},
                    "violations": {"type": ["object", "null"], "additionalProperties": True},
                    "entity_links": {
                        "type": ["object", "null"],
                        "properties": {
                            "subj_wikidata": {"type": ["string", "null"]},
                            "obj_wikidata": {"type": ["string", "null"]},
                        },
                        "additionalProperties": False,
                    },
                    "render_hint": {"type": ["string", "null"]},
                    # Pronoun/alias tolerance fields
                    "has_unresolved_pronoun": {"type": ["boolean", "null"]},
                    "original_subject": {"type": ["string", "null"]},
                    "subject_source": {"type": ["string", "null"]},
                    "subject_confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
                    "alias_map": {"type": ["object", "null"]},
                    "pronoun_subj": {"type": ["boolean", "null"]},
                    "pronoun_obj": {"type": ["boolean", "null"]},
                    "entities": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                    # Canonical evidence and anchor fields
                    "evidence_canonical": {"type": ["string", "null"], "minLength": 4},
                    "anchor_entity": {"type": ["string", "null"], "minLength": 1},
                    "lead_in_note_id": {"type": ["string", "null"], "minLength": 1},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    },
}

ALLOWED_TYPES = ["PERSON", "WORK", "ORG", "PLACE", "EVENT", "CONCEPT", "TIME"]

# 简化版：用于生成阶段的 Guided JSON/Schema 约束，字段少、约束简单，生成后仍会用 NOTE_JSON_SCHEMA 做严格校验。
NOTE_GEN_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "subj": {"type": "string", "minLength": 1},
            "pred": {"type": "string", "minLength": 1},
            "obj": {"type": "string", "minLength": 1},
            "subj_type": {"type": "string", "enum": ALLOWED_TYPES},
            "obj_type": {"type": "string", "enum": ALLOWED_TYPES},
            "evidence": {"type": "string", "minLength": 4},
            "meta": {
                "type": "object",
                "properties": {
                    "source": {"type": ["string", "null"]},
                    "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
                    "subject_profile": {"type": ["object", "null"]},
                    "attribute": {"type": ["object", "null"]},
                },
                "additionalProperties": True,
            },
        },
        "required": ["subj", "pred", "obj", "subj_type", "obj_type", "evidence", "meta"],
        "additionalProperties": False,
    },
}

# 默认允许的谓词集合；若存在 schema/predicates.json 则以文件为准（小写化）
ALLOWED_PREDICATES = [
    "performed_by",
    "authored_by",
    "directed_by",
    "spouse",
    "parent",
    "born_in",
    "located_in",
    "member_of",
    "works_for",
    "affiliated_with",
    "acted_in",
    "produced_by",
    "released_in",
    "label",
    "founded_by",
    "founded_on",
    "headquartered_in",
    "winner_of",
    "part_of",
    "occupation",
    "title",
    "category",
    "nationality",
    "born_on",
    "died_on",
    "alias_of",
    "same_as",
    "type",
    "has_member_count",
    "has_species_count",
]

# 尝试从配置文件加载替换
# (Moved to after PRED_SYNONYM_SETS definition)

# 属性映射表：谓词到规范化属性名，用于索引层归一
PRED2ATTR = {
    "occupation": "occupation",
    "profession": "occupation",
    "professions": "occupation",
    "job": "occupation",
    "jobs": "occupation",
    "works as": "occupation",
    "works_as": "occupation",
    "career": "occupation",
    "careers": "occupation",
    "founded": "founded_on",
    "founded_on": "founded_on",
    "founded in": "founded_on",
    "founded_in": "founded_on",
    "established": "founded_on",
    "established_on": "founded_on",
    "established in": "founded_on",
    "established_in": "founded_on",
    "year founded": "founded_on",
    "founded year": "founded_on",
    "works_for": "works_for",
    "worked for": "works_for",
    "worked_for": "works_for",
    "employed by": "works_for",
    "employed_by": "works_for",
    "employer": "works_for",
    "affiliated_with": "affiliated_with",
    "affiliated with": "affiliated_with",
    "professor at": "affiliated_with",
    "served as professor at": "affiliated_with",
    "taught at": "affiliated_with",
    "teaches at": "affiliated_with",
    "faculty at": "affiliated_with",
    "worked at": "affiliated_with",
    "works at": "affiliated_with",
    "member count": "has_member_count",
    "number of members": "has_member_count",
    "members": "has_member_count",
    "has members": "has_member_count",
    "contains members": "has_member_count",
    "member total": "has_member_count",
    "species": "has_species_count",
    "species count": "has_species_count",
    "number of species": "has_species_count",
    "has species": "has_species_count",
    "contains species": "has_species_count",
}

PRED_SYNONYM_SETS = {
    "performed_by": {"recorded_by", "artist", "performed_by"},
    "authored_by": {"written_by", "authored_by", "author_of", "wrote"},
    "directed_by": {"directed_by", "director_of", "directed", "who_directed"},
    "spouse": {"married_to", "partner", "spouse", "spouse_of", "wife_of", "husband_of"},
    "parent": {"father", "mother", "parent", "parent_of"},
    "born_in": {"place_of_birth", "born_in", "born_at", "native_of"},
    "acted_in": {"starring", "cast_in", "acted_in", "starred_in", "played", "portrayed", "starred", "features", "featuring"},
    "located_in": {"located_in"},
    "produced_by": {"produced_by"},
    "released_in": {"released_in"},
    "label": {"label"},
    "member_of": {"member_of"},
    "works_for": {"works_for", "worked_for", "employed_by", "works for", "worked for", "employed by", "employer"},
    "affiliated_with": {
        "affiliated_with",
        "affiliated with",
        "professor at",
        "served as professor at",
        "taught at",
        "teaches at",
        "faculty at",
        "worked at",
        "works at",
    },
    "founded_by": {"founded_by"},
    "founded_on": {"founded_on", "founded", "founded_in", "founded_on", "established", "established_in", "established_on"},
    "headquartered_in": {"headquartered_in"},
    "winner_of": {"winner_of"},
    "part_of": {"part_of"},
    "has_member_count": {
        "has_member_count",
        "member count",
        "number of members",
        "members",
        "has members",
        "contains members",
        "consists of members",
        "member total",
    },
    "has_species_count": {
        "has_species_count",
        "species count",
        "number of species",
        "species",
        "has species",
        "contains species",
        "genus of species",
    },
    # 强制归一：职业相关同义词全部归并到 "occupation"
    "occupation": {
        "occupation",
        "occupations",
        "profession",
        "professions",
        "job",
        "jobs",
        "works as",
        "works_as",
        "career",
        "careers",
        "title (when occupational)",
    },
    "title": {"title", "position", "role", "titles"},
    "category": {"category", "categories", "classification"},
    "nationality": {"nationality", "citizenship", "country_of_citizenship", "heritage", "ethnicity", "ethnic_origin"},
    "born_on": {"born_on", "birth_date", "date_of_birth", "born", "born on"},
    "died_on": {"died_on", "death_date", "date_of_death", "died"},
    "alias_of": {"alias_of", "aka", "also_known_as"},
    "same_as": {"same_as", "identical_to"},
    "type": {"type", "entity_type", "category_type"},
}

# 尝试从配置文件加载替换
try:
    _PRED_PATH = Path(__file__).resolve().parents[1] / "schema" / "predicates.json"
    if _PRED_PATH.exists():
        with open(_PRED_PATH, "r", encoding="utf-8") as fh:
            _preds = json.load(fh)
            if isinstance(_preds, list) and _preds:
                ALLOWED_PREDICATES = [str(p).strip().lower() for p in _preds if str(p).strip()]
            elif isinstance(_preds, dict) and _preds:
                # 若为字典，则 key 为 allowed predicates，values 为同义词扩充
                ALLOWED_PREDICATES = [str(p).strip().lower() for p in _preds.keys() if str(p).strip()]
                # 更新 PRED_SYNONYM_SETS (merge strategy: overwrite or append?)
                # 这里采用 overwrite 策略：配置文件定义的覆盖代码内置
                for canon, aliases in _preds.items():
                    canon = str(canon).strip().lower()
                    if not canon:
                        continue
                    if isinstance(aliases, list):
                        new_set = {str(a).strip().lower() for a in aliases if str(a).strip()}
                        # 确保 canon 自身也在集合中 (虽然后面 logic 会 handle，但保持一致性)
                        new_set.add(canon)
                        PRED_SYNONYM_SETS[canon] = new_set
except Exception:
    # 若加载失败，保留默认集合
    pass

CANONICAL_PREDICATES = set(ALLOWED_PREDICATES)


def canonicalize_predicate(raw_pred: str | None, raw_text: str | None = None) -> str | None:
    """Map surface predicate strings to canonical predicates."""
    raw = (raw_pred or "").strip().lower()
    raw_text = (raw_text or "").strip().lower()
    if not raw and not raw_text:
        return None
    if raw_text:
        if re.search(r"\b(professor|taught at|teaches at|faculty at|affiliated with)\b", raw_text):
            return "affiliated_with"
    candidates = {raw} if raw else set()
    if raw:
        candidates.add(raw.replace("_", " "))
        candidates.add(raw.replace(" ", "_"))
    for candidate in candidates:
        for canon, synonyms in PRED_SYNONYM_SETS.items():
            if candidate == canon or candidate in synonyms:
                return canon
        mapped = PRED2ATTR.get(candidate)
        if mapped:
            return mapped
    if raw_text:
        if "species" in raw_text and re.search(r"\b\d+\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", raw_text):
            return "has_species_count"
        if "member" in raw_text and re.search(r"\b\d+\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", raw_text):
            return "has_member_count"
    return raw or None
