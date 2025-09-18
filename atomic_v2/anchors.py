import re, string
_CAP_SEQ = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_QUOTES  = re.compile(r'[\"""]([^\"""]{3,60})[\"""]')
_DATE    = re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4})\b")
_NUMBER  = re.compile(r"\b\d{2,}\b")
_TRIVIAL = {"green","blue","album","season","series","lake","party","politics"}

def _norm_ws(s:str)->str: return re.sub(r"\s+"," ",s.strip())
def normalize_anchor(a:str)->str:
    a=_norm_ws(a.lower()); return a.translate(str.maketrans("","",string.punctuation))

def extract_anchors(text:str)->list[str]:
    if not text: return []
    a=[]
    a += [_norm_ws(m.group(1)) for m in _QUOTES.finditer(text)]
    a += [_norm_ws(m.group(1)) for m in _CAP_SEQ.finditer(text)]
    a += list(_DATE.findall(text)) + list(_NUMBER.findall(text))
    a=[normalize_anchor(x) for x in a if x]
    return [x for x in a if len(x)>=3 and x not in _TRIVIAL]

def anchor_overlap(a1:set[str], a2:set[str])->tuple[int,float]:
    inter=a1&a2
    if not inter: return 0,0.0
    w=sum(len(x) for x in inter); maxw=sum(len(x) for x in (a1|a2)) or 1
    return len(inter), min(1.0, w/maxw)