from typing import List,Dict,Any
from atomic_v2.anchors import extract_anchors, normalize_anchor, anchor_overlap
from atomic_v2.sketch_lsh import char_jaccard
from graph_v2.struct_prior import struct_score

def _conf():
    return {"topB":20,"jacc_char_min":0.18,"anchors_min_shared":2,
            "mutual_knn":{"enabled":False,"topK":50,"m":2,"cos_min":0.45}}

def build_candidates_for_note(center_id:int,notes:List[Dict[str,Any]],weak_graph,config:Dict[str,Any]|None=None)->List[Dict[str,Any]]:
    cfg={**_conf(),**(config or {})}; topB=int(cfg["topB"])
    jmin=float(cfg["jacc_char_min"]); amin=int(cfg["anchors_min_shared"])
    id2={n["id"]:n for n in notes}; c=id2.get(center_id)
    if not c: return []
    ctxt=f'{c.get("title","")} {c.get("text","")}'; ca=set(map(normalize_anchor,extract_anchors(ctxt)))
    out=[]
    for nid,n in id2.items():
        if nid==center_id: continue
        t=f'{n.get("title","")} {n.get("text","")}'
        j=char_jaccard(ctxt,t); a=set(map(normalize_anchor,extract_anchors(t)))
        cnt,ascore=anchor_overlap(ca,a); s=struct_score(center_id,nid,weak_graph)
        if (j>=jmin) or (cnt>=amin) or (s>0):
            score=0.35*j+0.25*ascore+0.25*s
            out.append({"id":nid,"score":float(score),"reasons":{"jacc":j,"anchors":cnt,"struct":s,"mknn":0}})
    out.sort(key=lambda x:x["score"],reverse=True)
    return out[:topB]