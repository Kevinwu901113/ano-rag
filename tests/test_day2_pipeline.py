import json
from pathlib import Path
from retrieval_v2.candidate_gate import build_candidates_for_note
from graph_v2.struct_prior import build_weak_graph
from atomic_v2.anchors import extract_anchors
from atomic_v2.dbtes import dbtes_select_edges
from retrieval_v2.mmr import mmr_select
from graph_v2.materialize import write_snapshot

DATA = json.loads(Path("tests/data/musique_sample.json").read_text(encoding="utf-8"))
PARAS = {p["idx"]: p for p in DATA["paragraphs"]}

def _note(idx):
    p = PARAS[idx]
    return {"id": p["idx"], "title": p["title"], "text": p["paragraph_text"], "embedding": None}

def mock_llm_call(prompt: str) -> str:
    # 轻量规则：如果出现 "Steve Hillage" 与某候选同现，则选它；否则选第一条
    import re
    lines = [l for l in prompt.splitlines() if l.startswith("CANDIDATE")]
    for ln in lines:
        if ("Steve Hillage".lower() in ln.lower()) and ("Miquette Giraudy".lower() in ln.lower()):
            return "WIN: " + ln.split(":",1)[1].strip()
    return "WIN: " + (lines[0].split(":",1)[1].strip() if lines else "NONE")

def test_dbtes_picks_correct_edge():
    notes = [_note(i) for i in PARAS]
    Gweak = build_weak_graph(PARAS)
    cands = build_candidates_for_note(10, notes, Gweak, {"topB": 20, "jacc_char_min":0.18, "anchors_min_shared":2, "mutual_knn":{"enabled":False}})
    # 只保留 id 与 gate 分数，模拟真实输入
    cands = [{"id": c["id"], "score": c["score"]} for c in cands]
    sel = dbtes_select_edges(center_note=_note(10), candidates=cands, llm_call=mock_llm_call,
                             config={"m":6,"rounds":2,"keep_k":3,"max_tokens_per_call":300}, notes=notes)
    ids = {e["id"] for e in sel}
    assert 5 in ids, "DBTES 应该把 #5 选出来"
    # 选出的每条边需携带简短理由
    assert all("reason" in e and isinstance(e["reason"], str) and len(e["reason"])>0 for e in sel)

def test_mmr_and_materialize_snapshot(tmp_path):
    # 构造一个小集合，MMR 选 <= keep_k
    items = [{"id":5, "score":0.9, "vec":[0.9,0.1]}, {"id":0, "score":0.7, "vec":[0.8,0.2]}, {"id":8, "score":0.6, "vec":[0.81,0.19]}]
    chosen = mmr_select(items, k=2, lambda_=0.7, sim=lambda a,b: sum(x*y for x,y in zip(a["vec"], b["vec"])))
    assert len(chosen) == 2
    # 落图
    edges = [{"src":10, "dst":x["id"], "weight":x["score"], "evidence":{"reason":"test"}} for x in chosen]
    out = write_snapshot(edges=edges, notes=[_note(i) for i in PARAS], out_dir=str(tmp_path), degree_cap=20)
    assert Path(out, "graph.graphml").exists()
    assert Path(out, "edges.jsonl").exists()
