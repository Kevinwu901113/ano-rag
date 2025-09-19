import os, ast, re, json, sys
ROOT=os.getcwd()
ENTRY=["scripts/build_note_graph_v2.py","tests/test_day1_pipeline.py","tests/test_day2_pipeline.py","main.py"]
KEEP_PREFIX=("atomic_v2/","graph_v2/","retrieval_v2/","llm/","scripts/","tests/","config.yaml",".github/","CODEOWNERS")
IGNORE={".git",".venv","venv","__pycache__", ".pytest_cache",".mypy_cache"}

def rel(p): return os.path.relpath(p, ROOT).replace("\\","/")
def list_py():
  out=[]
  for d,dirs,files in os.walk(ROOT):
    if os.path.basename(d) in IGNORE: dirs[:]=[]; continue
    for f in files:
      if f.endswith(".py"): out.append(rel(os.path.join(d,f)))
  return out

def mods_from(p):
  try: src=open(p,"r",encoding="utf-8").read()
  except: return set()
  try: t=ast.parse(src,filename=p)
  except: return set()
  mods=set()
  for n in ast.walk(t):
    if isinstance(n, ast.Import):
      for a in n.names: mods.add(a.name.split(".")[0])
    elif isinstance(n, ast.ImportFrom):
      if n.module: mods.add(n.module.split(".")[0])
  return mods

def idx(py):
  m={}
  for p in py:
    mod=p[:-12] if p.endswith("/__init__.py") else p[:-3]
    mod=mod.replace("/",".")
    top=mod.split(".")[0]
    m.setdefault(top,[]).append(p)
  return m

def resolve(entry, py):
  mod=idx(py); used=set(); stack=[e for e in entry if os.path.exists(e)]
  while stack:
    p=stack.pop()
    if p in used: continue
    used.add(p)
    for top in mods_from(p):
      for path in mod.get(top,[]):
        if path not in used: stack.append(path)
  return used

py=list_py(); used=resolve(ENTRY,py)
dead=[p for p in py if p not in used and not p.startswith(KEEP_PREFIX)]
prio=[p for p in dead if re.search(r"(old|legacy|v1|backup|bak|temp|tmp)", p, re.I)]
norm=[p for p in dead if p not in prio]

json.dump({"priority":prio,"normal":norm}, open("dead_code_candidates.json","w"), ensure_ascii=False, indent=2)
print("Saved dead_code_candidates.json")