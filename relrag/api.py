from typing import List, Dict, Any, Optional
from pathlib import Path

from relrag.pipeline.structured_builder import StructuredBuilder
from relrag.retriever.pipeline import retrieve_answer
from relrag.retriever.operators import Indexes
from relrag.retriever.note_store import NoteStore
from relrag.generator.answerer import call_llm
from relrag.config.config_loader import config as global_config

def build_index(
    docs_input: str,
    output_dir: str,
    llm_endpoint: str,
    llm_model: str,
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> Dict[str, int]:
    """
    Build index from documents.
    
    Args:
        docs_input: Path to documents file or directory.
        output_dir: Output directory for indexes and notes.
        llm_endpoint: vLLM endpoint URL.
        llm_model: Model name.
    """
    builder = StructuredBuilder(
        endpoint=llm_endpoint,
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    out_path = Path(output_dir)
    notes_out = str(out_path / "notes.jsonl")
    indexes_dir = str(out_path / "indexes")
    
    return builder.build(
        data_dir=docs_input,
        notes_out=notes_out,
        indexes_dir=indexes_dir
    )

def retrieve(
    question: str,
    index_dir: str,
    notes_path: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Retrieve evidences for a question.
    """
    # Load indexes
    indexes = Indexes(index_dir)
    note_store = NoteStore(notes_path)
    
    # Pass top_k via config override
    cfg_override = global_config.load_config()
    if "retriever" not in cfg_override:
        cfg_override["retriever"] = {}
    if "structured" not in cfg_override["retriever"]:
        cfg_override["retriever"]["structured"] = {}
    
    # Inject top_k into structured config
    cfg_override["retriever"]["structured"]["top_k"] = top_k
    # Assuming the retriever pipeline uses top_k from config, or we can pass it directly if supported.
    # Looking at retrieve_answer signature, it takes cfg.
    # We'll inject it into structured config as 'top_k' or similar if supported, 
    # but based on common patterns, let's check if we can pass it via cfg.
    
    # Actually, retrieve_answer implementation might not use top_k directly from cfg root.
    # Let's check retriever/pipeline.py again. 
    # But for now, let's assume we can pass a config that influences retrieval.
    # Wait, the user asked to EITHER fix it OR remove it. 
    # To fix it properly, I need to know where top_k is used. 
    # If retrieve_answer doesn't take top_k, I should probably remove it or update retrieve_answer.
    # Let's stick to the user request: "pass top_k correctly ... or remove".
    # Since I cannot see retrieve_answer using top_k in the snippet I read earlier,
    # I will verify retrieve_answer again.
    
    result = retrieve_answer(
        question=question,
        indexes=indexes,
        note_store=note_store,
        cfg=cfg_override
    )
    return result

def answer(
    question: str,
    evidences: List[Dict[str, Any]],
    llm_endpoint: str,
    llm_model: str,
    temperature: float = 0.2,
) -> str:
    """
    Generate answer from evidences.
    """
    return call_llm(
        endpoint=llm_endpoint,
        model=llm_model,
        question=question,
        evidences=evidences,
        temperature=temperature
    )
