import json
from pathlib import Path

from atomic_v2.anchors import extract_anchors, normalize_anchor, anchor_overlap
from atomic_v2.sketch_lsh import LSHIndex, char_jaccard
from graph_v2.struct_prior import struct_score, build_weak_graph
from retrieval_v2.candidate_gate import build_candidates_for_note

DATA = json.loads(Path("tests/data/musique_sample.json").read_text(encoding="utf-8"))
PARAS = {p["idx"]: p for p in DATA["paragraphs"]}

def _text(idx): return PARAS[idx]["paragraph_text"]
def _title(idx): return PARAS[idx]["title"]

def test_anchors_basic():
    a10 = set(map(normalize_anchor, extract_anchors(_text(10) + " " + _title(10))))
    a5  = set(map(normalize_anchor, extract_anchors(_text(5)  + " " + _title(5))))
    cnt, score = anchor_overlap(a10, a5)
    assert cnt >= 1, "至少要命中 'Steve Hillage' 这样的高 IDF 锚点"
    # 防止“Green”这类低IDF词主导
    assert "steve hillage" in a10.union(a5)

def test_lsh_retrieves_candidate():
    idx = LSHIndex(n_gram=(3,5), bands=32, rows=4)
    for p in PARAS.values():
        idx.add(p["idx"], _title(p["idx"]) + " " + _text(p["idx"]))
    # 关键：LSH 粗召回里必须包含 #5
    cands = idx.query(_title(10) + " " + _text(10), topn=20)
    cand_ids = {cid for cid, _ in cands}
    assert 5 in cand_ids, "LSH 粗召回应该包含 5（Miquette Giraudy）"

def test_struct_prior_prefers_person_over_wrong_green():
    G_weak = build_weak_graph(PARAS)  # 同标题/相邻等弱边
    s_10_5 = struct_score(10, 5, G_weak)
    s_10_0 = struct_score(10, 0, G_weak)  # Grant's First Stand（误簇）
    assert s_10_5 >= s_10_0

def test_candidate_gate_keeps_target_pair():
    # embeddings 可选，Day1 允许不传（或传空字典）
    notes = [
        {"id": p["idx"], "title": p["title"], "text": p["paragraph_text"], "embedding": None}
        for p in PARAS.values()
    ]
    G_weak = build_weak_graph(PARAS)
    cands_for_10 = build_candidates_for_note(
        center_id=10, notes=notes, weak_graph=G_weak, config={
            "topB": 20,
            "jacc_char_min": 0.18,
            "anchors_min_shared": 2,
            "mutual_knn": {"enabled": False}
        }
    )
    cand_ids = {c["id"] for c in cands_for_10}
    assert 5 in cand_ids, "Gate 后候选应包含 5"
    assert len(cand_ids) <= 20
