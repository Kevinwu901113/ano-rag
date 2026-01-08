from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    logger.warning("pandas unavailable: {}. Embedding index disabled.", exc)

from relrag.config import config as config_loader
from relrag.config.config_loader import DEFAULT_EMBED_MODEL
from relrag.utils.text_builders import build_note_text_for_embed
try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore
    logger.warning("FAISS is not available: {}", exc)


class EmbeddingIndexBuilder:
    """Build/extend FAISS index over notes.jsonl."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or config_loader.load_config()
        self.notes_path = Path(self.cfg.get("notes", {}).get("out_path", "notes/notes.jsonl"))
        retriever_cfg = self.cfg.get("retriever", {})
        self.embed_cfg = retriever_cfg.get("embedding", {})

    def build(self) -> None:
        if not self.embed_cfg.get("enabled", False):
            logger.info("Embedding retriever disabled; skip FAISS build.")
            return
        if faiss is None or pd is None:
            logger.warning("Embedding index build skipped: faiss/pandas unavailable.")
            return
        if not self.notes_path.exists():
            raise FileNotFoundError(f"Notes file not found: {self.notes_path}")

        offline_path = Path(self.embed_cfg.get("offline_index_path", "indexes/faiss/notes.faiss"))
        meta_path = Path(self.embed_cfg.get("meta_path", "indexes/faiss/notes.meta.parquet"))
        offline_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        provider = self.embed_cfg.get("provider", "qwen3")
        model = self._resolve_model_name()
        cache_dir = self._clean_path(self.embed_cfg.get("cache_dir"))
        device = self._resolve_device()
        dtype = self.embed_cfg.get("dtype")
        try:
            from relrag.utils.embedding_utils import get_shared_encoder
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Embedding encoder unavailable: {}. Skip FAISS build.", exc)
            return
        encoder = get_shared_encoder(
            provider,
            model,
            max_length=int(self.embed_cfg.get("max_len_note", 256)),
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
        )

        existing_meta = self._load_existing_meta(meta_path)
        seen_note_ids = set(existing_meta["note_id"]) if existing_meta is not None else set()

        new_texts: List[str] = []
        new_meta: List[Dict[str, Any]] = []
        with self.notes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                note = json.loads(line)
                note_id = note.get("note_id")
                if not note_id or note_id in seen_note_ids:
                    continue
                text = build_note_text_for_embed(note, max_len=self.embed_cfg.get("max_len_note"))
                if not text:
                    continue
                new_texts.append(text)
                new_meta.append(self._note_meta(note))
        if not new_texts:
            logger.info("No new notes to index for embeddings.")
            return

        vectors = encoder.encode(new_texts)
        if vectors.size == 0:
            logger.warning("No vectors produced for embedding index.")
            return
        if bool(self.embed_cfg.get("normalize", True)):
            faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = self._load_or_create_index(offline_path, dim)
        if not index.is_trained:
            logger.info("Training FAISS index ({}) on {} vectors", self.embed_cfg.get("faiss", {}).get("kind", "HNSW32"), len(vectors))
            index.train(vectors)
        start_id = index.ntotal
        index.add(vectors)
        faiss.write_index(index, str(offline_path))

        self._persist_meta(meta_path, new_meta, start_id)
        logger.info(
            "Embedding index updated: +{} vectors (total={}) -> {}",
            len(new_meta),
            start_id + len(new_meta),
            offline_path,
        )

    def _load_existing_meta(self, meta_path: Path) -> Optional[pd.DataFrame]:
        if pd is None:
            return None
        if not meta_path.exists():
            return None
        try:
            return pd.read_parquet(meta_path)
        except Exception as exc:
            logger.warning("Failed to read existing embedding meta: {}", exc)
            return None

    def _note_meta(self, note: Dict[str, Any]) -> Dict[str, Any]:
        meta = (note.get("meta") or {}) if isinstance(note, dict) else {}
        return {
            "note_id": note.get("note_id"),
            "subj": note.get("subj"),
            "pred": note.get("pred"),
            "obj": note.get("obj"),
            "subj_type": note.get("subj_type"),
            "obj_type": note.get("obj_type"),
            "final_conf": meta.get("final_conf"),
            "quality_score": meta.get("quality_score"),
            "domain": meta.get("domain"),
        }

    def _persist_meta(self, meta_path: Path, new_meta: List[Dict[str, Any]], start_id: int) -> None:
        df_new = pd.DataFrame(new_meta)
        df_new.insert(0, "vector_id", [start_id + idx for idx in range(len(df_new))])
        existing = self._load_existing_meta(meta_path)
        combined = pd.concat([existing, df_new], ignore_index=True) if existing is not None else df_new
        combined.sort_values("vector_id", inplace=True)
        combined.to_parquet(meta_path, index=False)

    def _load_or_create_index(self, path: Path, dim: int):
        kind = str((self.embed_cfg.get("faiss") or {}).get("kind", "HNSW32")).upper()
        if path.exists():
            logger.info("Loading existing FAISS index: {}", path)
            return faiss.read_index(str(path))
        if kind.startswith("HNSW"):
            m = int(kind.replace("HNSW", "") or 32)
            index = faiss.IndexHNSWFlat(dim, m)
            ef_c = int((self.embed_cfg.get("faiss") or {}).get("efSearch", 128))
            index.hnsw.efConstruction = max(ef_c, 128)
            return index
        if kind.startswith("IVF") and "PQ" in kind:
            # kind format: IVF4096,PQ64
            ivf_part, pq_part = kind.split(",", 1)
            nlist = int(ivf_part.replace("IVF", "") or 4096)
            m = int(pq_part.replace("PQ", "") or 64)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
            return index
        logger.warning("Unknown faiss.kind '{}', falling back to IndexFlatIP.", kind)
        return faiss.IndexFlatIP(dim)

    def _clean_path(self, value: Any) -> Optional[str]:
        if not value:
            return None
        return str(Path(str(value)).expanduser())

    def _resolve_model_name(self) -> str:
        override = self.embed_cfg.get("model_path_override")
        base = self.embed_cfg.get("model", DEFAULT_EMBED_MODEL)
        candidate = str(override or base).strip()
        if not candidate:
            raise ValueError("Embedding model name is not configured")
        if override:
            logger.info("Embedding model override detected: {}", candidate)
        return candidate

    def _resolve_device(self) -> Optional[str]:
        device = self.embed_cfg.get("device")
        if device:
            return str(device)
        system_cfg = self.cfg.get("system") or {}
        return system_cfg.get("device")


def main() -> None:
    cfg = config_loader.load_config()
    builder = EmbeddingIndexBuilder(cfg)
    builder.build()


if __name__ == "__main__":
    main()
