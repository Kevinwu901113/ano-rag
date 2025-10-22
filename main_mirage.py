#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIRAGE Benchmark Runner for AnoRAG System

This script runs the AnoRAG system on the MIRAGE benchmark in Mixed mode,
generating answers from a global document pool and ensuring output compatibility
with MIRAGE's official evaluation.py without performing evaluation itself.

Key Features:
- Support for Mixed/Oracle/Base modes
- Global document pool indexing (BM25/Dense/Hybrid)
- Optional atomic note extraction and graph building
- Note-to-original-chunk remapping for evaluation compatibility
- Parallel processing with resume capability
- Comprehensive logging and reproducibility

Usage:
    python main_mirage.py --mode mixed --topk 5 --retriever hybrid
    python main_mirage.py --mode oracle --new --debug
    python main_mirage.py --mode base --model qwen2.5-7b --temperature 0.1
    python main_mirage.py --mode mixed --embed-model sentence-transformers/all-MiniLM-L6-v2
"""

import os
import sys
import json
import time
import hashlib
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core imports
from config import config
from utils.file_utils import FileUtils
from utils.text_utils import TextUtils
from utils.logging_utils import setup_logging
from vector_store import VectorRetriever, EmbeddingManager
from llm.local_llm import LocalLLM
from llm.lmstudio_client import LMStudioClient
from llm.atomic_note_generator import AtomicNoteGenerator
from llm.enhanced_atomic_note_generator import EnhancedAtomicNoteGenerator
from llm.parallel_task_atomic_note_generator import ParallelTaskAtomicNoteGenerator
from query.query_processor import QueryProcessor
from MIRAGE.utils import load_json, convert_doc_pool
from graph import GraphBuilder, GraphIndex

# Constants
MIRAGE_DATA_DIR = "mirage"
RESULT_DIR = "result"
DEFAULT_TOPK = 5
DEFAULT_MAX_WORKERS_QUERY = 4
DEFAULT_MAX_WORKERS_NOTE = 2

@dataclass
class MirageConfig:
    """Configuration for MIRAGE run"""
    # Run configuration
    run_id: str
    mode: str  # mixed, oracle, base
    topk: int
    new_run: bool
    debug: bool
    
    # Data paths
    dataset_path: str
    doc_pool_path: str
    oracle_path: str
    result_dir: str
    
    # Retrieval configuration
    retriever_type: str  # bm25, dense, hybrid
    embed_model: str
    rebuild_index: bool
    
    # LLM configuration
    model_name: str
    temperature: float
    max_tokens: int
    seed: Optional[int]
    
    # Note engine configuration
    note_engines: List[str]
    enable_notes: bool
    enable_graph: bool
    
    # Parallel configuration
    max_workers_query: int
    max_workers_note: int
    
    # Timing
    start_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

class MirageRunner:
    """Main runner for MIRAGE benchmark"""
    
    def __init__(self, config: MirageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths
        self.setup_paths()
        
        # Data storage
        self.dataset: List[Dict[str, Any]] = []
        self.doc_pool: List[Dict[str, Any]] = []
        self.oracle_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # Components
        self.retriever: Optional[VectorRetriever] = None
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.llm_client: Optional[Union[LocalLLM, LMStudioClient]] = None
        self.note_generator: Optional[AtomicNoteGenerator] = None
        self.query_processor: Optional[QueryProcessor] = None
        self.atomic_notes: List[Dict[str, Any]] = []

        # Doc pool indexes for reliable remapping
        self.doc_pool_index_by_chunk_id: Dict[str, int] = {}
        self.doc_pool_index_by_hash: Dict[str, List[int]] = {}
        self.doc_pool_offset_index: Dict[Tuple[str, int, int], int] = {}
        self._doc_chunk_embeddings: Optional[Any] = None
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'processed_queries': 0,
            'failed_queries': 0,
            'total_time': 0.0,
            'index_build_time': 0.0,
            'note_generation_time': 0.0,
            'retrieval_time': 0.0,
            'generation_time': 0.0,
            'graph_build_time': 0.0
        }
    
    def setup_paths(self):
        """Setup directory structure"""
        self.run_dir = Path(self.config.result_dir) / self.config.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "index").mkdir(exist_ok=True)
        (self.run_dir / "notes").mkdir(exist_ok=True)
        (self.run_dir / "graph").mkdir(exist_ok=True)
        (self.run_dir / "queries").mkdir(exist_ok=True)
        
        self.logger.info(f"Run directory: {self.run_dir}")
    
    def load_data(self) -> bool:
        """Load MIRAGE dataset, doc_pool, and oracle data"""
        try:
            self.logger.info("Loading MIRAGE data...")
            
            # Load dataset
            self.dataset = load_json(self.config.dataset_path)
            self.logger.info(f"Loaded {len(self.dataset)} queries from dataset")
            
            # Load doc_pool - it's already in the correct format as a list
            self.doc_pool = load_json(self.config.doc_pool_path)
            self.logger.info(f"Loaded {len(self.doc_pool)} documents from doc_pool")
            self._build_doc_pool_indexes()

            # Load oracle data if needed
            if self.config.mode == "oracle":
                self.oracle_data = load_json(self.config.oracle_path)
                self.logger.info(f"Loaded oracle data for {len(self.oracle_data)} queries")
            
            self.stats['total_queries'] = len(self.dataset)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return False
    
    def build_global_index(self) -> bool:
        """Build or load global retrieval index"""
        try:
            self.logger.info("Building global retrieval index...")
            start_time = time.time()
            
            # Initialize embedding manager when vector retrieval is required or custom model provided
            if self.config.retriever_type in ['dense', 'hybrid'] or self.config.embed_model:
                self.embedding_manager = EmbeddingManager()
                if self.config.embed_model:
                    self.embedding_manager.set_model(self.config.embed_model)
            else:
                self.embedding_manager = None

            # Initialize retriever with configured components
            self.retriever = VectorRetriever(embedding_manager=self.embedding_manager,
                                             retrieval_mode=self.config.retriever_type)
            if self.config.embed_model:
                self.retriever.set_embedding_model(self.config.embed_model)

            # Set index directory to run-specific location
            index_dir = self.run_dir / "index"
            self.retriever.data_dir = str(index_dir)

            # Prefer cached/generated atomic notes for indexing
            atomic_notes = self._prepare_atomic_notes_for_index()

            # Build index
            success = self.retriever.build_index(
                atomic_notes=atomic_notes,
                force_rebuild=self.config.rebuild_index,
                save_index=True
            )
            
            if not success:
                self.logger.error("Failed to build retrieval index")
                return False
            
            self.stats['index_build_time'] = time.time() - start_time
            self.logger.info(f"Index built successfully in {self.stats['index_build_time']:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {e}")
            return False

    def _prepare_atomic_notes_for_index(self) -> List[Dict[str, Any]]:
        """Return the best available atomic notes for index construction."""
        if self.atomic_notes:
            self.logger.info(
                "Using %d in-memory atomic notes for index build",
                len(self.atomic_notes)
            )
            return self.atomic_notes

        from_doc_pool = self._collect_atomic_notes_from_doc_pool()
        if from_doc_pool:
            self.logger.info(
                "Using %d atomic notes from doc_pool for index build",
                len(from_doc_pool)
            )
            self.atomic_notes = from_doc_pool
            return from_doc_pool

        cached_notes = self._load_cached_atomic_notes()
        if cached_notes:
            self.logger.info(
                "Using %d cached atomic notes for index build",
                len(cached_notes)
            )
            self.atomic_notes = cached_notes
            return cached_notes

        fallback_notes = self.convert_doc_pool_to_notes()
        self.logger.info(
            "No atomic notes available; falling back to %d doc_pool chunks",
            len(fallback_notes)
        )
        self.atomic_notes = fallback_notes
        return fallback_notes

    def convert_doc_pool_to_notes(self) -> List[Dict[str, Any]]:
        """Convert doc_pool entries to atomic notes format for indexing"""
        atomic_notes = []

        for idx, doc in enumerate(self.doc_pool):
            chunk_id, doc_hash, offsets = self._prepare_doc_metadata(doc, idx)
            note = {
                'note_id': chunk_id,
                'content': doc.get('doc_chunk', ''),
                'doc_name': doc.get('doc_name', ''),
                'chunk_id': chunk_id,
                'doc_hash': doc_hash,
                'offsets': offsets,
                'original_index': idx,
                'metadata': {
                    'source': 'doc_pool',
                    'mapped_id': doc.get('mapped_id'),
                    'support': doc.get('support', False),
                    'chunk_id': chunk_id,
                    'doc_hash': doc_hash,
                    'offsets': offsets,
                    'doc_name': doc.get('doc_name', ''),
                    'original_index': idx
                }
            }
            atomic_notes.append(note)

        return atomic_notes

    def _prepare_doc_metadata(self, doc: Dict[str, Any], index: int) -> Tuple[str, str, List[int]]:
        """Ensure doc chunk metadata is normalized and return identifiers."""
        chunk_id = self._extract_chunk_id(doc, index)
        doc_chunk = doc.get('doc_chunk', '') or ''
        doc_hash = doc.get('doc_hash') or self._compute_doc_hash(doc_chunk)

        offsets = doc.get('offsets')
        if not self._is_valid_offset(offsets):
            offsets = [0, len(doc_chunk)]

        # Persist normalized metadata back to doc for downstream use
        doc['chunk_id'] = chunk_id
        doc['doc_hash'] = doc_hash
        doc['offsets'] = list(offsets)

        return chunk_id, doc_hash, list(offsets)

    @staticmethod
    def _compute_doc_hash(content: str) -> str:
        return hashlib.sha1((content or '').encode()).hexdigest()

    @staticmethod
    def _is_valid_offset(offsets: Any) -> bool:
        return isinstance(offsets, (list, tuple)) and len(offsets) == 2 and all(
            isinstance(x, (int, float)) for x in offsets
        )

    def _extract_chunk_id(self, doc: Dict[str, Any], index: int) -> str:
        chunk_id = (
            doc.get('chunk_id')
            or doc.get('metadata', {}).get('chunk_id')
            or f"{doc.get('doc_name', f'doc_{index}')}#{index}"
        )
        return str(chunk_id)

    def _build_doc_pool_indexes(self) -> None:
        """Create indexes to quickly locate doc_pool chunks by metadata."""
        self.doc_pool_index_by_chunk_id.clear()
        self.doc_pool_index_by_hash.clear()
        self.doc_pool_offset_index.clear()
        self._doc_chunk_embeddings = None

        for idx, doc in enumerate(self.doc_pool):
            chunk_id, doc_hash, offsets = self._prepare_doc_metadata(doc, idx)

            if chunk_id:
                self.doc_pool_index_by_chunk_id[chunk_id] = idx

            if doc_hash:
                self.doc_pool_index_by_hash.setdefault(doc_hash, []).append(idx)

            if self._is_valid_offset(offsets):
                doc_name = doc.get('doc_name', '')
                self.doc_pool_offset_index[(doc_name, int(offsets[0]), int(offsets[1]))] = idx

    def _ensure_doc_pool_indexes(self) -> None:
        if not self.doc_pool_index_by_chunk_id and self.doc_pool:
            self._build_doc_pool_indexes()

    def _lookup_by_offsets(self, doc_name: str, offsets: Optional[List[int]]) -> Optional[int]:
        if not self._is_valid_offset(offsets):
            return None

        offset_tuple = (doc_name or '', int(offsets[0]), int(offsets[1]))
        return self.doc_pool_offset_index.get(offset_tuple)

    def _verify_doc_match(
        self,
        doc: Dict[str, Any],
        expected_chunk_id: Optional[str],
        expected_hash: Optional[str],
        expected_offsets: Optional[List[int]]
    ) -> bool:
        if expected_chunk_id and str(doc.get('chunk_id')) != str(expected_chunk_id):
            return False

        if expected_hash:
            doc_hash = doc.get('doc_hash') or self._compute_doc_hash(doc.get('doc_chunk', ''))
            if doc_hash != expected_hash:
                return False

        if self._is_valid_offset(expected_offsets):
            doc_offsets = doc.get('offsets')
            if not self._is_valid_offset(doc_offsets) or [int(doc_offsets[0]), int(doc_offsets[1])] != [
                int(expected_offsets[0]), int(expected_offsets[1])
            ]:
                return False

        return True

    def _ensure_doc_embeddings(self) -> Optional[Any]:
        if self._doc_chunk_embeddings is not None:
            return self._doc_chunk_embeddings

        if not self.retriever or not getattr(self.retriever, 'embedding_manager', None):
            return None

        contents = [doc.get('doc_chunk', '') or '' for doc in self.doc_pool]
        if not contents:
            return None

        try:
            embeddings = self.retriever.embedding_manager.encode_texts(contents)
            self._doc_chunk_embeddings = embeddings
            return embeddings
        except Exception as exc:  # pragma: no cover - safety net
            if self.config.debug:
                self.logger.debug(f"Failed to encode doc_pool chunks for embeddings: {exc}")
            return None

    def _find_chunk_by_embedding(self, content: str) -> Optional[Tuple[Dict[str, Any], int]]:
        if not content or not content.strip():
            return None

        embeddings = self._ensure_doc_embeddings()
        if embeddings is None:
            return None

        try:
            note_embedding = self.retriever.embedding_manager.encode_texts([content])
        except Exception as exc:  # pragma: no cover - safety net
            if self.config.debug:
                self.logger.debug(f"Failed to encode note content for embedding remap: {exc}")
            return None

        try:
            import numpy as np

            doc_matrix = np.array(embeddings, dtype=float)
            note_vector = np.array(note_embedding, dtype=float)

            if note_vector.ndim == 1:
                note_vector = note_vector.reshape(1, -1)

            if doc_matrix.ndim == 1:
                doc_matrix = doc_matrix.reshape(len(self.doc_pool), -1)

            if doc_matrix.size == 0 or note_vector.size == 0:
                return None

            note_norm = np.linalg.norm(note_vector, axis=1, keepdims=True)
            doc_norm = np.linalg.norm(doc_matrix, axis=1, keepdims=False)

            if not note_norm.size or not doc_norm.size or float(note_norm.max()) == 0.0:
                return None

            similarities = (doc_matrix @ note_vector.T).reshape(-1) / (
                (doc_norm * note_norm.squeeze()) + 1e-8
            )

            if similarities.size == 0:
                return None

            best_idx = int(similarities.argmax())
            best_score = float(similarities[best_idx])

            if best_score < 0.6:
                return None

            return self.doc_pool[best_idx], best_idx
        except Exception as exc:  # pragma: no cover - numerical safety
            if self.config.debug:
                self.logger.debug(f"Embedding remap failed: {exc}")
            return None

    @staticmethod
    def _get_chunk_key(doc: Dict[str, Any], index: int) -> str:
        return str(doc.get('chunk_id') or f"{doc.get('doc_name', '')}#{index}")
    
    def generate_atomic_notes(self) -> bool:
        """Generate atomic notes from doc_pool (optional)"""
        if not self.config.enable_notes:
            # Still attempt to harvest any pre-existing notes for downstream steps
            self.atomic_notes = self._collect_atomic_notes_from_doc_pool()
            return True

        try:
            self.logger.info("Generating atomic notes...")
            start_time = time.time()

            # Reuse existing notes if they are already attached to the doc_pool
            existing_notes = self._collect_atomic_notes_from_doc_pool()
            if existing_notes:
                self.logger.info(
                    "Found %d existing atomic notes; skipping regeneration",
                    len(existing_notes)
                )
                self.atomic_notes = existing_notes
                self._persist_atomic_notes(existing_notes)
                return True

            # Initialize LLM for note generation
            llm = LocalLLM()

            # Prefer parallel task generator when enabled in config
            try:
                self.note_generator = ParallelTaskAtomicNoteGenerator(
                    llm=llm,
                    max_workers=self.config.max_workers_note
                )
                self.logger.info("Using ParallelTaskAtomicNoteGenerator")
            except Exception:
                # Fallback to enhanced or baseline generators
                try:
                    self.note_generator = EnhancedAtomicNoteGenerator(
                        llm=llm,
                        max_workers=self.config.max_workers_note
                    )
                    self.logger.info("Using EnhancedAtomicNoteGenerator")
                except Exception:
                    self.note_generator = AtomicNoteGenerator(
                        llm=llm,
                        max_workers=self.config.max_workers_note
                    )
                    self.logger.info("Using AtomicNoteGenerator")
            
            # Convert doc_pool to text_chunks format
            text_chunks = []
            for idx, doc in enumerate(self.doc_pool):
                chunk = {
                    'text': doc.get('doc_chunk', ''),
                    'chunk_index': idx,
                    'source_info': {
                        'title': doc.get('doc_name', ''),
                        'document_id': doc.get('mapped_id', f'doc_{idx}'),
                        'is_supporting': doc.get('support', False)
                    }
                }
                text_chunks.append(chunk)
            
            # Generate notes
            notes = self.note_generator.generate_atomic_notes(text_chunks)
            self.logger.info(f"Generated {sum(len(n.get('notes', [])) for n in notes)} notes across {len(text_chunks)} chunks")

            # Store notes back in doc_pool for downstream use
            for idx, doc in enumerate(self.doc_pool):
                note_result = notes[idx] if idx < len(notes) else {}
                raw_notes = note_result.get('notes', []) if isinstance(note_result, dict) else []
                normalized_notes = []
                for note_idx, note in enumerate(raw_notes):
                    normalized = self._normalize_atomic_note(
                        note,
                        doc,
                        doc_index=idx,
                        note_position=len(normalized_notes)
                    )
                    if normalized:
                        normalized_notes.append(normalized)
                doc['notes'] = normalized_notes

            self.stats['note_generation_time'] = time.time() - start_time
            self.logger.info(f"Atomic notes generation completed in {self.stats['note_generation_time']:.2f}s")

            self.atomic_notes = self._collect_atomic_notes_from_doc_pool()
            if self.atomic_notes:
                self._persist_atomic_notes(self.atomic_notes)
            return True

        except Exception as e:
            self.logger.error(f"Failed to generate atomic notes: {e}")
            return False

    def _normalize_atomic_note(
        self,
        note: Dict[str, Any],
        doc: Optional[Dict[str, Any]] = None,
        *,
        doc_index: Optional[int] = None,
        note_position: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Normalize a single atomic note structure for downstream compatibility."""
        if not isinstance(note, dict):
            return None

        normalized = json.loads(json.dumps(note)) if note else {}
        metadata = dict(normalized.get('metadata') or {})

        chunk_id = normalized.get('chunk_id') or metadata.get('chunk_id')
        doc_hash = normalized.get('doc_hash') or metadata.get('doc_hash')
        offsets = normalized.get('offsets') or metadata.get('offsets')
        doc_name = normalized.get('doc_name') or metadata.get('doc_name')
        original_index = normalized.get('original_index', metadata.get('original_index'))

        if doc is not None and doc_index is not None and doc_index >= 0:
            chunk_id, doc_hash, offsets = self._prepare_doc_metadata(doc, doc_index)
            doc_name = doc.get('doc_name', doc_name or '')
            original_index = doc_index

        if chunk_id is None:
            chunk_id = f"chunk_{doc_index if doc_index is not None else 'na'}_{note_position}"

        if doc_hash is None:
            text_for_hash = (
                normalized.get('content')
                or normalized.get('raw_span')
                or (doc.get('doc_chunk') if doc else '')
                or ''
            )
            doc_hash = self._compute_doc_hash(text_for_hash)

        if not self._is_valid_offset(offsets):
            base_text = normalized.get('content') or normalized.get('raw_span') or ''
            offsets = [0, len(base_text)]

        note_id = normalized.get('note_id') or normalized.get('id')
        if not note_id:
            base_doc = doc_name or (doc.get('doc_name') if doc else 'doc')
            note_id = f"{base_doc}#{chunk_id}#note_{note_position}"

        content = (
            normalized.get('content')
            or normalized.get('raw_span')
            or normalized.get('summary')
            or (doc.get('doc_chunk') if doc else '')
            or ''
        )

        resolved_doc_name = doc_name or ''
        if doc is not None:
            resolved_doc_name = resolved_doc_name or doc.get('doc_name', '')

        normalized.update(
            {
                'note_id': str(note_id),
                'chunk_id': str(chunk_id),
                'doc_name': resolved_doc_name,
                'doc_hash': doc_hash,
                'offsets': list(offsets),
                'original_index': original_index if original_index is not None else doc_index,
                'content': content,
            }
        )

        metadata.update(
            {
                'note_id': normalized['note_id'],
                'chunk_id': normalized['chunk_id'],
                'doc_hash': normalized['doc_hash'],
                'doc_name': normalized['doc_name'],
                'offsets': normalized['offsets'],
                'original_index': normalized.get('original_index'),
            }
        )
        normalized['metadata'] = metadata

        if doc is not None:
            source_info = doc.get('source_info') or {
                'title': doc.get('doc_name', ''),
                'document_id': doc.get('mapped_id'),
                'is_supporting': doc.get('support', False)
            }
            normalized.setdefault('source_info', source_info)

        return normalized

    def _collect_atomic_notes_from_doc_pool(self) -> List[Dict[str, Any]]:
        """Harvest normalized atomic notes from the current doc_pool."""
        if not self.doc_pool:
            return []

        self._ensure_doc_pool_indexes()

        collected: List[Dict[str, Any]] = []
        for doc_index, doc in enumerate(self.doc_pool):
            notes = doc.get('notes')
            if not notes:
                doc['notes'] = []
                continue

            if not isinstance(notes, list):
                notes = [notes]

            normalized_notes = []
            for note_position, note in enumerate(notes):
                normalized = self._normalize_atomic_note(
                    note,
                    doc,
                    doc_index=doc_index,
                    note_position=note_position
                )
                if normalized:
                    normalized_notes.append(normalized)
                    collected.append(normalized)
            doc['notes'] = normalized_notes

        return collected

    def _match_doc_for_note(self, note: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """Find the doc_pool entry corresponding to a cached note."""
        if not self.doc_pool:
            return None, None

        self._ensure_doc_pool_indexes()

        metadata = note.get('metadata', {}) or {}
        chunk_id = note.get('chunk_id') or metadata.get('chunk_id')
        doc_hash = note.get('doc_hash') or metadata.get('doc_hash')
        offsets = note.get('offsets') or metadata.get('offsets')
        doc_name = note.get('doc_name') or metadata.get('doc_name') or ''
        original_index = note.get('original_index', metadata.get('original_index'))

        if isinstance(original_index, str):
            try:
                original_index = int(original_index)
            except ValueError:
                original_index = None

        if original_index is not None and 0 <= original_index < len(self.doc_pool):
            candidate = self.doc_pool[original_index]
            if self._verify_doc_match(candidate, chunk_id, doc_hash, offsets):
                return candidate, original_index

        if chunk_id:
            idx = self.doc_pool_index_by_chunk_id.get(str(chunk_id))
            if idx is not None:
                candidate = self.doc_pool[idx]
                if self._verify_doc_match(candidate, chunk_id, doc_hash, offsets):
                    return candidate, idx

        if doc_hash:
            for idx in self.doc_pool_index_by_hash.get(doc_hash, []):
                candidate = self.doc_pool[idx]
                if doc_name and candidate.get('doc_name') != doc_name:
                    continue
                if self._verify_doc_match(candidate, None, doc_hash, offsets):
                    return candidate, idx

        offset_idx = self._lookup_by_offsets(doc_name, offsets)
        if offset_idx is not None and 0 <= offset_idx < len(self.doc_pool):
            return self.doc_pool[offset_idx], offset_idx

        return None, None

    def _load_cached_atomic_notes(self) -> List[Dict[str, Any]]:
        """Load atomic notes from disk and align them with the doc_pool."""
        notes_path = self.run_dir / "notes" / "atomic_notes.jsonl"
        if not notes_path.exists():
            return []

        loaded: List[Dict[str, Any]] = []
        with open(notes_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded.append(json.loads(line))
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse cached atomic note line: {line[:80]}")

        if not loaded:
            return []

        normalized: List[Dict[str, Any]] = []
        for note in loaded:
            doc, doc_index = self._match_doc_for_note(note)
            normalized_note = self._normalize_atomic_note(
                note,
                doc,
                doc_index=doc_index,
                note_position=len(normalized)
            )
            if normalized_note:
                normalized.append(normalized_note)
                if doc is not None and doc_index is not None and doc_index >= 0:
                    notes_list = self.doc_pool[doc_index].setdefault('notes', [])
                    if normalized_note not in notes_list:
                        notes_list.append(normalized_note)

        return normalized

    def _persist_atomic_notes(self, atomic_notes: List[Dict[str, Any]]) -> None:
        """Write atomic notes to disk for reuse across runs."""
        if not atomic_notes:
            return

        notes_dir = self.run_dir / "notes"
        notes_dir.mkdir(exist_ok=True)
        notes_path = notes_dir / "atomic_notes.jsonl"

        with open(notes_path, 'w', encoding='utf-8') as handle:
            for note in atomic_notes:
                handle.write(json.dumps(note, ensure_ascii=False) + '\n')

    def build_graph(self) -> bool:
        """Build and save knowledge graph (optional)"""
        if not self.config.enable_graph:
            return True

        # Prefer generated atomic notes; fall back to doc_pool conversion
        if hasattr(self, 'atomic_notes') and self.atomic_notes:
            notes_for_graph = self.atomic_notes
        else:
            self.logger.warning("No generated atomic notes; falling back to doc_pool-based notes for graph")
            notes_for_graph = self.convert_doc_pool_to_notes()
            if not notes_for_graph:
                self.logger.warning("No notes available from doc_pool; skipping graph build")
                return True
            
        try:
            self.logger.info("Building knowledge graph...")
            start_time = time.time()
            
            # Initialize GraphBuilder without LLM (LLM-enhanced relations disabled pre-LLM init)
            graph_builder = GraphBuilder(llm=self.llm_client)
            
            # Compute embeddings for notes
            if not self.embedding_manager:
                self.embedding_manager = EmbeddingManager()
            
            note_embeddings = self.embedding_manager.encode_atomic_notes(notes_for_graph)
            
            # Build graph
            graph = graph_builder.build_graph(notes_for_graph, embeddings=note_embeddings)
            
            # Index graph
            graph_index = GraphIndex()
            graph_index.build_index(graph, atomic_notes=notes_for_graph, embeddings=note_embeddings)
            
            # Save graph artifacts
            graph_dir = self.run_dir / "graph"
            graph_dir.mkdir(exist_ok=True)
            
            graph_file = graph_dir / "graph.json"
            graph_index.save_index(str(graph_file))
            graph_index.save_graphml(str(graph_dir / "graph.graphml"))

            # Cache path for downstream use if needed
            self.graph_file = str(graph_file)

            self.stats['graph_build_time'] = time.time() - start_time
            self.logger.info(f"Graph built and saved in {self.stats['graph_build_time']:.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"Failed to build graph: {e}")
            return False

    def initialize_llm(self) -> bool:
        """Initialize LLM client for answer generation"""
        try:
            self.logger.info(f"Initializing LLM: {self.config.model_name}")
            
            # Prefer LM Studio client; fall back to LocalLLM only if instantiation fails
            try:
                self.llm_client = LMStudioClient(model=self.config.model_name)
                self.logger.info("Using LM Studio client")
            except Exception as e:
                self.logger.warning(f"LM Studio client init failed, falling back: {e}")
                self.llm_client = LocalLLM(
                    model_name=self.config.model_name,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                self.logger.info("Using LocalLLM client")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            return False
    
    def process_single_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single query through the full pipeline"""
        query_id = query_data['query_id']
        query = query_data['query']
        
        try:
            self.logger.info(f"Processing query {query_id}: {query[:50]}...")
            
            # Create query directory
            query_dir = self.run_dir / "queries" / str(query_id)
            query_dir.mkdir(exist_ok=True)
            
            # Check if already processed
            done_flag = query_dir / "done.flag"
            if done_flag.exists() and not self.config.debug:
                self.logger.info(f"Query {query_id} already processed, skipping")
                # Load existing result
                contexts_file = query_dir / "contexts.json"
                answer_file = query_dir / "answer.txt"
                
                if contexts_file.exists() and answer_file.exists():
                    contexts = FileUtils.read_json(str(contexts_file))
                    answer = FileUtils.read_file(str(answer_file))
                    return {
                        'id': query_id,
                        'predicted_answer': answer.strip(),
                        'retrieved_contexts': contexts,
                        'error': False
                    }
            
            start_time = time.time()
            
            # Step 1: Retrieval
            contexts = self.retrieve_contexts(query_data, query_dir)
            
            # Step 2: Answer generation
            answer = self.generate_answer(query, contexts, query_dir)
            
            # Step 3: Save results
            result = {
                'id': query_id,
                'predicted_answer': answer,
                'retrieved_contexts': contexts,
                'error': False
            }
            
            # Save timing
            timing = {
                'total_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            FileUtils.write_json(timing, str(query_dir / "timing.json"))
            
            # Mark as done
            done_flag.touch()
            
            self.logger.info(f"Query {query_id} processed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process query {query_id}: {e}")
            return {
                'id': query_id,
                'predicted_answer': "",
                'retrieved_contexts': [],
                'error': True
            }

    def save_predictions(self, results: List[Dict[str, Any]]):
        """Save predictions to JSONL format"""
        predictions_file = self.run_dir / "predictions.jsonl"

        # Clean results for output (remove error field)
        clean_results = []
        for result in results:
            clean_result = {
                'id': result['id'],
                'predicted_answer': result['predicted_answer'],
                'retrieved_contexts': result['retrieved_contexts']
            }
            clean_results.append(clean_result)

        predictions_file.unlink(missing_ok=True)
        for clean_result in clean_results:
            FileUtils.append_jsonl_atomic(str(predictions_file), clean_result)
        self.logger.info(f"Predictions saved to {predictions_file}")

    def save_manifest(self):
        """Save run manifest with configuration and metadata"""
        manifest = {
            'run_id': self.config.run_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'data_info': {
                'dataset_path': self.config.dataset_path,
                'dataset_size': len(self.dataset),
                'dataset_hash': self.calculate_file_hash(self.config.dataset_path),
                'dataset_file_size_bytes': FileUtils.get_file_size_bytes(self.config.dataset_path),
                'dataset_line_count': FileUtils.count_file_lines(self.config.dataset_path),
                'doc_pool_path': self.config.doc_pool_path,
                'doc_pool_size': len(self.doc_pool),
                'doc_pool_hash': self.calculate_file_hash(self.config.doc_pool_path),
                'doc_pool_file_size_bytes': FileUtils.get_file_size_bytes(self.config.doc_pool_path),
                'doc_pool_line_count': FileUtils.count_file_lines(self.config.doc_pool_path),
            },
            'statistics': self.stats,
            'output_files': {
                'predictions': 'predictions.jsonl',
                'logs': ['logs/run.log', 'logs/run_error.log'],
                'manifest': 'manifest.json'
            }
        }

        if self.config.mode == "oracle":
            manifest['data_info']['oracle_path'] = self.config.oracle_path
            manifest['data_info']['oracle_hash'] = self.calculate_file_hash(self.config.oracle_path)
            manifest['data_info']['oracle_file_size_bytes'] = FileUtils.get_file_size_bytes(self.config.oracle_path)
            manifest['data_info']['oracle_line_count'] = FileUtils.count_file_lines(self.config.oracle_path)
        
        # Add notes and graph artifacts if present
        notes_path = self.run_dir / "notes" / "atomic_notes.jsonl"
        if notes_path.exists():
            manifest['output_files']['notes'] = str(notes_path.relative_to(self.run_dir))
        graph_json = self.run_dir / "graph" / "graph.json"
        if graph_json.exists():
            manifest['output_files']['graph'] = {
                'index': 'graph/graph.json',
                'embeddings': 'graph/graph_embeddings.npz',
                'mappings': 'graph/graph_mappings.json',
                'graphml': 'graph/graph.graphml'
            }
        
        manifest_file = self.run_dir / "manifest.json"
        FileUtils.write_manifest(str(manifest_file), manifest)
        self.logger.info(f"Manifest saved to {manifest_file}")

    def retrieve_contexts(self, query_data: Dict[str, Any], query_dir: Path) -> List[Dict[str, str]]:
        """Retrieve contexts based on mode"""
        query_id = query_data['query_id']
        query = query_data['query']
        
        if self.config.mode == "base":
            # Base mode: no contexts
            contexts = []
        elif self.config.mode == "oracle":
            # Oracle mode: use gold standard contexts
            oracle_entry = self.oracle_data.get(str(query_id))
            contexts = []
            if oracle_entry:
                # Oracle data is a single entry, not a list
                contexts.append({
                    'title': oracle_entry.get('doc_name', ''),
                    'text': oracle_entry.get('doc_chunk', '')
                })
            else:
                self.logger.warning(f"No oracle data found for query {query_id}")
        else:
            # Mixed mode: retrieve from global doc_pool
            contexts = self.retrieve_mixed_mode(query, query_dir)
        
        # Save contexts
        FileUtils.write_json(contexts, str(query_dir / "contexts.json"))
        
        return contexts

    def retrieve_mixed_mode(self, query: str, query_dir: Path) -> List[Dict[str, str]]:
        """Retrieve contexts in mixed mode"""
        try:
            # Perform retrieval
            results = self.retriever.search([query], top_k=self.config.topk * 2)[0]  # Get more candidates for better remapping
            
            # Save detailed retrieval results if debug mode
            if self.config.debug:
                retrieval_data = {
                    'query': query,
                    'mode': self.config.mode,
                    'topk': self.config.topk,
                    'results': []
                }
                
                for result in results:
                    retrieval_data['results'].append({
                        'score': result.get('score', 0.0),
                        'note_id': result.get('note_id', ''),
                        'content': result.get('content', ''),
                        'retrieval_method': result.get('retrieval_method', 'unknown'),
                        'original_index': result.get('original_index')
                    })
                
                FileUtils.write_json(retrieval_data, str(query_dir / "retrieval.json"))
            
            # Convert results to contexts (remap notes to original chunks)
            contexts = self.remap_notes_to_chunks(results)
            
            return contexts
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve contexts: {e}")
            return []

    def remap_notes_to_chunks(self, retrieval_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Remap retrieved notes back to original doc_pool chunks"""
        contexts = []
        seen_chunks = set()

        self._ensure_doc_pool_indexes()

        for result in retrieval_results:
            try:
                resolved = self._resolve_chunk_from_result(result)

                if not resolved:
                    if self.config.debug:
                        self.logger.debug(
                            f"Failed to resolve note '{result.get('note_id')}' back to doc chunk"
                        )
                    continue

                doc, idx = resolved
                chunk_key = self._get_chunk_key(doc, idx)

                if chunk_key in seen_chunks:
                    if self.config.debug:
                        self.logger.debug(
                            f"Skipping duplicate chunk {chunk_key} for note {result.get('note_id')}"
                        )
                    continue

                contexts.append({
                    'title': doc.get('doc_name', ''),
                    'text': doc.get('doc_chunk', '')
                })
                seen_chunks.add(chunk_key)

                if len(contexts) >= self.config.topk:
                    break

            except Exception as e:
                self.logger.warning(f"Failed to remap result: {e}")
                continue
        
        # Debug logging
        if self.config.debug:
            mapping_info = {
                'total_results': len(retrieval_results),
                'mapped_contexts': len(contexts),
                'target_topk': self.config.topk
            }
            mapping_file = self.run_dir / "notes" / "mapping.jsonl"
            mapping_file.parent.mkdir(exist_ok=True)
            
            with open(mapping_file, 'a') as f:
                f.write(json.dumps(mapping_info) + '\n')

        return contexts[:self.config.topk]

    def _resolve_chunk_from_result(self, result: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], int]]:
        metadata = result.get('metadata', {}) or {}
        note_id = result.get('note_id') or metadata.get('note_id') or ''
        chunk_id = result.get('chunk_id') or metadata.get('chunk_id') or note_id
        doc_hash = result.get('doc_hash') or metadata.get('doc_hash')
        offsets = result.get('offsets') or metadata.get('offsets')
        doc_name = result.get('doc_name') or metadata.get('doc_name')

        original_index = result.get('original_index', metadata.get('original_index'))
        if isinstance(original_index, str):
            try:
                original_index = int(original_index)
            except ValueError:
                original_index = None

        # 1. direct index mapping
        if original_index is not None and 0 <= original_index < len(self.doc_pool):
            doc = self.doc_pool[original_index]
            if self._verify_doc_match(doc, chunk_id, doc_hash, offsets):
                return doc, original_index

        # 2. chunk_id mapping (with metadata verification)
        if chunk_id:
            idx = self.doc_pool_index_by_chunk_id.get(str(chunk_id))
            if idx is not None:
                doc = self.doc_pool[idx]
                if self._verify_doc_match(doc, chunk_id, doc_hash, offsets):
                    return doc, idx

        # 3. doc_hash mapping
        if doc_hash:
            candidate_indexes = self.doc_pool_index_by_hash.get(doc_hash, [])
            for idx in candidate_indexes:
                doc = self.doc_pool[idx]
                if doc_name and doc.get('doc_name') != doc_name:
                    continue
                if self._verify_doc_match(doc, None, doc_hash, offsets):
                    return doc, idx

        # 4. offsets mapping
        offset_idx = self._lookup_by_offsets(doc_name, offsets)
        if offset_idx is not None and 0 <= offset_idx < len(self.doc_pool):
            return self.doc_pool[offset_idx], offset_idx

        # 5. embedding-based fallback
        embedding_match = self._find_chunk_by_embedding(result.get('content', ''))
        if embedding_match:
            return embedding_match

        # 6. substring fallback
        content = result.get('content', '')
        if content:
            best_match = self.find_best_chunk_match(content)
            if best_match and 'index' in best_match:
                return best_match, best_match['index']

        return None
    
    def find_best_chunk_match(self, content: str) -> Optional[Dict[str, Any]]:
        """Find best matching chunk in doc_pool using content similarity"""
        if not content:
            return None
        
        best_match = None
        best_score = 0.0
        
        # Simple substring matching as fallback
        for idx, doc in enumerate(self.doc_pool):
            doc_chunk = doc.get('doc_chunk', '')
            if not doc_chunk:
                continue
            
            # Calculate simple overlap score
            if content in doc_chunk:
                score = len(content) / len(doc_chunk)
                if score > best_score:
                    best_score = score
                    best_match = {**doc, 'index': idx}
            elif doc_chunk in content:
                score = len(doc_chunk) / len(content)
                if score > best_score:
                    best_score = score
                    best_match = {**doc, 'index': idx}
        
        return best_match if best_score > 0.5 else None
    
    def generate_answer(self, query: str, contexts: List[Dict[str, str]], query_dir: Path) -> str:
        """Generate answer using LLM"""
        try:
            # Prepare context string
            context_str = ""
            if contexts:
                for i, ctx in enumerate(contexts, 1):
                    context_str += f"{i}. {ctx['text']}\n\n"
            else:
                context_str = "No context provided."
            
            # Generate answer using appropriate method
            if hasattr(self.llm_client, 'generate_final_answer'):
                # Use specialized final answer generation method
                answer = self.llm_client.generate_final_answer(context_str.strip(), query)
            else:
                # Fallback to generic generate method with proper prompt
                system_prompt = "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly."
                
                if context_str.strip() == "No context provided.":
                    prompt = f"Question: {query}\n\nAnswer:"
                else:
                    prompt = f"Context:\n{context_str.strip()}\n\nQuestion: {query}\n\nAnswer:"
                
                answer = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            
            # Clean answer
            answer = answer.strip() if answer else ""
            
            # Handle empty or invalid answers
            if not answer or answer.lower() in ['', 'none', 'n/a', 'unknown']:
                answer = "I don't know."
            
            # Normalize answer using TextUtils
            normalized_answer = TextUtils.normalize_text(answer)
            
            # Save both versions
            FileUtils.write_file(str(query_dir / "answer.txt"), answer)
            FileUtils.write_file(str(query_dir / "answer_norm.txt"), normalized_answer)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Failed to generate answer: {e}")
            # Return empty answer on failure
            FileUtils.write_file(str(query_dir / "answer.txt"), "")
            FileUtils.write_file(str(query_dir / "answer_norm.txt"), "")
            return ""
    
    def run_parallel_processing(self) -> List[Dict[str, Any]]:
        """Run parallel processing of all queries"""
        results = []
        completed_queries = set()
        
        # Check for existing completed queries
        for query_data in self.dataset:
            query_id = query_data['query_id']
            query_dir = self.run_dir / "queries" / str(query_id)
            done_flag = query_dir / "done.flag"
            
            if done_flag.exists():
                completed_queries.add(query_id)
                # Load existing result
                try:
                    contexts_file = query_dir / "contexts.json"
                    answer_file = query_dir / "answer.txt"
                    
                    if contexts_file.exists() and answer_file.exists():
                        contexts = FileUtils.read_json(str(contexts_file))
                        answer = FileUtils.read_file(str(answer_file))
                        results.append({
                            'id': query_id,
                            'predicted_answer': answer.strip(),
                            'retrieved_contexts': contexts,
                            'error': False
                        })
                        self.stats['processed_queries'] += 1
                    else:
                        # Mark as incomplete if files are missing
                        completed_queries.discard(query_id)
                        done_flag.unlink(missing_ok=True)
                except Exception as e:
                    self.logger.warning(f"Failed to load existing result for query {query_id}: {e}")
                    completed_queries.discard(query_id)
                    done_flag.unlink(missing_ok=True)
        
        # Filter out completed queries
        remaining_queries = [q for q in self.dataset if q['query_id'] not in completed_queries]
        
        if completed_queries:
            self.logger.info(f"Found {len(completed_queries)} completed queries, processing {len(remaining_queries)} remaining")
        
        if not remaining_queries:
            self.logger.info("All queries already completed")
            # Sort results by query_id to maintain order
            results.sort(key=lambda x: x['id'])
            return results
        
        # Process remaining queries in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers_query) as executor:
            # Submit remaining queries
            future_to_query = {
                executor.submit(self.process_single_query, query_data): query_data
                for query_data in remaining_queries
            }
            
            # Collect results with progress tracking
            for future in as_completed(future_to_query):
                query_data = future_to_query[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['error']:
                        self.stats['failed_queries'] += 1
                    else:
                        self.stats['processed_queries'] += 1
                    
                    # Log progress
                    total_processed = self.stats['processed_queries'] + self.stats['failed_queries']
                    self.logger.info(f"Progress: {total_processed}/{self.stats['total_queries']} queries processed")
                    
                except Exception as e:
                    self.logger.error(f"Query {query_data['query_id']} failed: {e}")
                    results.append({
                        'id': query_data['query_id'],
                        'predicted_answer': "",
                        'retrieved_contexts': [],
                        'error': True
                    })
                    self.stats['failed_queries'] += 1
        
        # Sort results by query_id to maintain order
        results.sort(key=lambda x: x['id'])
        return results
    
    def save_predictions(self, results: List[Dict[str, Any]]):
        """Save predictions to JSONL format"""
        predictions_file = self.run_dir / "predictions.jsonl"

        # Clean results for output (remove error field)
        clean_results = []
        for result in results:
            clean_result = {
                'id': result['id'],
                'predicted_answer': result['predicted_answer'],
                'retrieved_contexts': result['retrieved_contexts']
            }
            clean_results.append(clean_result)

        predictions_file.unlink(missing_ok=True)
        for clean_result in clean_results:
            FileUtils.append_jsonl_atomic(str(predictions_file), clean_result)
        self.logger.info(f"Predictions saved to {predictions_file}")
    
    def save_manifest(self):
        """Save run manifest with configuration and metadata"""
        manifest = {
            'run_id': self.config.run_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'data_info': {
                'dataset_path': self.config.dataset_path,
                'dataset_size': len(self.dataset),
                'dataset_hash': self.calculate_file_hash(self.config.dataset_path),
                'dataset_file_size_bytes': FileUtils.get_file_size_bytes(self.config.dataset_path),
                'dataset_line_count': FileUtils.count_file_lines(self.config.dataset_path),
                'doc_pool_path': self.config.doc_pool_path,
                'doc_pool_size': len(self.doc_pool),
                'doc_pool_hash': self.calculate_file_hash(self.config.doc_pool_path),
                'doc_pool_file_size_bytes': FileUtils.get_file_size_bytes(self.config.doc_pool_path),
                'doc_pool_line_count': FileUtils.count_file_lines(self.config.doc_pool_path),
            },
            'statistics': self.stats,
            'output_files': {
                'predictions': 'predictions.jsonl',
                'logs': ['logs/run.log', 'logs/run_error.log'],
                'manifest': 'manifest.json'
            }
        }

        if self.config.mode == "oracle":
            manifest['data_info']['oracle_path'] = self.config.oracle_path
            manifest['data_info']['oracle_hash'] = self.calculate_file_hash(self.config.oracle_path)
            manifest['data_info']['oracle_file_size_bytes'] = FileUtils.get_file_size_bytes(self.config.oracle_path)
            manifest['data_info']['oracle_line_count'] = FileUtils.count_file_lines(self.config.oracle_path)
        
        # Add notes and graph artifacts if present
        notes_path = self.run_dir / "notes" / "atomic_notes.jsonl"
        if notes_path.exists():
            manifest['output_files']['notes'] = str(notes_path.relative_to(self.run_dir))
        graph_json = self.run_dir / "graph" / "graph.json"
        if graph_json.exists():
            manifest['output_files']['graph'] = {
                'index': 'graph/graph.json',
                'embeddings': 'graph/graph_embeddings.npz',
                'mappings': 'graph/graph_mappings.json',
                'graphml': 'graph/graph.graphml'
            }
        
        manifest_file = self.run_dir / "manifest.json"
        FileUtils.write_manifest(str(manifest_file), manifest)
        self.logger.info(f"Manifest saved to {manifest_file}")
    
    def calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA1 hash of a file"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.sha1(f.read()).hexdigest()
        except Exception:
            return "unknown"
    
    def run(self) -> bool:
        """Main execution flow"""
        try:
            self.logger.info(f"Starting MIRAGE run: {self.config.run_id}")
            self.logger.info(f"Mode: {self.config.mode}, TopK: {self.config.topk}, Retriever: {self.config.retriever_type}")
            
            # Step 1: Load data
            if not self.load_data():
                return False
            
            # Step 2: Generate atomic notes (optional)
            if not self.generate_atomic_notes():
                return False

            # Step 3: Build global index (skip for base mode)
            if self.config.mode != "base":
                if not self.build_global_index():
                    return False

            # Step 3.5: Build graph (optional)
            if not self.build_graph():
                return False

            # Step 4: Initialize LLM
            if not self.initialize_llm():
                return False
            
            # Step 5: Process all queries
            self.logger.info("Starting query processing...")
            start_time = time.time()
            
            results = self.run_parallel_processing()
            
            self.stats['total_time'] = time.time() - start_time
            
            # Step 6: Save results
            self.save_predictions(results)
            self.save_manifest()
            
            # Step 7: Final statistics
            self.logger.info("=" * 50)
            self.logger.info("MIRAGE Run Completed")
            self.logger.info(f"Total queries: {self.stats['total_queries']}")
            self.logger.info(f"Processed: {self.stats['processed_queries']}")
            self.logger.info(f"Failed: {self.stats['failed_queries']}")
            self.logger.info(f"Total time: {self.stats['total_time']:.2f}s")
            self.logger.info(f"Results saved to: {self.run_dir}")
            self.logger.info("=" * 50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"MIRAGE run failed: {e}")
            return False

def generate_run_id(new_run: bool, result_dir: str) -> str:
    """Generate or find existing run ID"""
    if new_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"mirage_run_{timestamp}"

    # Find most recent run
    latest_run = FileUtils.get_latest_run_dir(result_dir, "mirage_run_")
    if latest_run:
        return latest_run.name

    # No existing runs, create new one
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"mirage_run_{timestamp}"

def parse_note_engines(engines_str: str) -> List[str]:
    """Parse note engines string into list"""
    if not engines_str:
        return []
    
    engines = []
    for engine in engines_str.split(','):
        engine = engine.strip()
        if engine:
            engines.append(engine)
    
    return engines

def build_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser for MIRAGE runs."""
    parser = argparse.ArgumentParser(
        description="Run AnoRAG system on MIRAGE benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_mirage.py --mode mixed --topk 5 --retriever hybrid
  python main_mirage.py --mode oracle --new --debug
  python main_mirage.py --mode base --model qwen2.5-7b --temperature 0.1
  python main_mirage.py --mode mixed --embed-model sentence-transformers/all-MiniLM-L6-v2
  python main_mirage.py --mode mixed --note_engines "ollama:qwen2.5-7b,lmstudio:qwen2.5-7b"
        """
    )

    # Mode and basic configuration
    parser.add_argument('--mode', choices=['mixed', 'oracle', 'base'], default='mixed',
                       help='Evaluation mode (default: mixed)')
    parser.add_argument('--topk', type=int, default=DEFAULT_TOPK,
                       help=f'Number of contexts per query (default: {DEFAULT_TOPK})')
    parser.add_argument('--new', action='store_true',
                       help='Create new run directory')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed logging')
    
    # Data paths
    parser.add_argument('--dataset', default=f'{MIRAGE_DATA_DIR}/dataset.json',
                       help='Path to dataset.json')
    parser.add_argument('--doc_pool', default=f'{MIRAGE_DATA_DIR}/doc_pool.json',
                       help='Path to doc_pool.json')
    parser.add_argument('--oracle', default=f'{MIRAGE_DATA_DIR}/oracle.json',
                       help='Path to oracle.json')
    parser.add_argument('--result_dir', default=RESULT_DIR,
                       help='Result directory')
    
    # Retrieval configuration
    parser.add_argument('--retriever', choices=['bm25', 'dense', 'hybrid'], default='hybrid',
                       help='Retriever type (default: hybrid)')
    parser.add_argument('--embed-model', '--embed_model', dest='embed_model', type=str,
                       help='Embedding model name (overrides config). Alias: --embed-model/--embed_model')
    parser.add_argument('--rebuild-index', '--rebuild_index', dest='rebuild_index', action='store_true',
                       help='Force rebuild index. Alias: --rebuild-index/--rebuild_index')
    
    # LLM configuration
    parser.add_argument('--model', default='openai/gpt-oss-20b',
                       help='LLM model name (default: openai/gpt-oss-20b)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='LLM temperature (default: 0.1)')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='LLM max tokens (default: 512)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    
    # Note generation configuration
    parser.add_argument('--note_engines', type=str,
                       help='Comma-separated list of note engines (e.g., "ollama:qwen2.5-7b,lmstudio:qwen2.5-7b")')
    parser.add_argument('--enable_notes', action='store_true',
                       help='Enable atomic note generation')
    parser.add_argument('--enable_graph', action='store_true',
                       help='Enable graph building and retrieval')
    
    # Parallel configuration
    parser.add_argument('--max_workers_query', type=int, default=DEFAULT_MAX_WORKERS_QUERY,
                       help=f'Max parallel query workers (default: {DEFAULT_MAX_WORKERS_QUERY})')
    parser.add_argument('--max_workers_note', type=int, default=DEFAULT_MAX_WORKERS_NOTE,
                       help=f'Max parallel note workers (default: {DEFAULT_MAX_WORKERS_NOTE})')
    
    return parser


def main(argv: Optional[List[str]] = None):
    """Main entry point"""
    parser = build_parser()

    args = parser.parse_args(argv)
    
    # Generate run ID
    run_id = generate_run_id(args.new, args.result_dir)
    
    # Create configuration
    mirage_config = MirageConfig(
        run_id=run_id,
        mode=args.mode,
        topk=args.topk,
        new_run=args.new,
        debug=args.debug,
        dataset_path=args.dataset,
        doc_pool_path=args.doc_pool,
        oracle_path=args.oracle,
        result_dir=args.result_dir,
        retriever_type=args.retriever,
        embed_model=args.embed_model or config.get('embedding.model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
        rebuild_index=args.rebuild_index,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        note_engines=parse_note_engines(args.note_engines),
        enable_notes=args.enable_notes or bool(args.note_engines),
        enable_graph=args.enable_graph,
        max_workers_query=args.max_workers_query,
        max_workers_note=args.max_workers_note,
        start_time=time.time()
    )
    
    # Setup logging
    log_dir = Path(args.result_dir) / run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(
        log_file=str(log_dir / "run.log"),
        log_level="DEBUG" if args.debug else "INFO"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting MIRAGE benchmark runner")
    logger.info(f"Configuration: {mirage_config.to_dict()}")
    
    # Create and run MIRAGE runner
    runner = MirageRunner(mirage_config)
    success = runner.run()
    
    if success:
        logger.info("MIRAGE run completed successfully")
        print(f"Results saved to: {Path(args.result_dir) / run_id}")
        print(f"Predictions file: {Path(args.result_dir) / run_id / 'predictions.jsonl'}")
        sys.exit(0)
    else:
        logger.error("MIRAGE run failed")
        sys.exit(1)

if __name__ == "__main__":
    main()