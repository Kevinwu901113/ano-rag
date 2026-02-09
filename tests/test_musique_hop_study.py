import math
import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "musique" / "hop_study.py"
SPEC = importlib.util.spec_from_file_location("hop_study_module", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
hop_study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hop_study
SPEC.loader.exec_module(hop_study)


def test_infer_hop_from_id_prefix_then_decomposition():
    rec = {"id": "4hop2__foo", "question_decomposition": [{"id": 1}, {"id": 2}]}
    hop, source = hop_study._infer_hop(rec)
    assert hop == 4
    assert source == "id_prefix"

    rec2 = {"id": "unknown_case", "question_decomposition": [{"id": 1}, {"id": 2}, {"id": 3}]}
    hop2, source2 = hop_study._infer_hop(rec2)
    assert hop2 == 3
    assert source2 == "decomposition_len"


def test_support_prf_basic_cases():
    gold = [("A", 1), ("B", 2)]
    pred = [("A", 1), ("C", 3)]
    p, r, f1 = hop_study._support_prf(gold, pred)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)

    p2, r2, f2 = hop_study._support_prf([], [])
    assert p2 == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)
    assert f2 == pytest.approx(1.0)


def test_fit_exp_returns_positive_decay_rate():
    fit = hop_study._fit_exp([(2, 0.8), (3, 0.5), (4, 0.3)])
    assert fit["n"] == 3
    assert fit["b"] > 0
    assert not math.isnan(fit["r2"])


def test_allocate_total_balanced_exact_target():
    alloc, effective, total_available = hop_study._allocate_total_balanced({2: 1271, 3: 763, 4: 425}, [2, 3, 4], 500)
    assert total_available == 2459
    assert effective == 500
    assert sum(alloc.values()) == 500
    assert alloc[2] == 167
    assert alloc[3] == 167
    assert alloc[4] == 166


def test_allocate_total_proportional_exact_target():
    alloc, effective, total_available = hop_study._allocate_total_proportional({2: 1252, 3: 760, 4: 405}, [2, 3, 4], 500)
    assert total_available == 2417
    assert effective == 500
    assert sum(alloc.values()) == 500
    assert alloc[2] == 259
    assert alloc[3] == 157
    assert alloc[4] == 84


@pytest.mark.skipif(hop_study.yaml is None, reason="PyYAML unavailable")
def test_build_relrag_no_walk_config(tmp_path: Path):
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text(
        "retriever:\n"
        "  structured:\n"
        "    enabled: true\n"
        "    walk_enabled: true\n",
        encoding="utf-8",
    )
    out_cfg = tmp_path / "nowalk.yaml"
    hop_study._build_relrag_no_walk_config(base_cfg, out_cfg)
    payload = hop_study.yaml.safe_load(out_cfg.read_text(encoding="utf-8"))
    assert payload["retriever"]["structured"]["walk_enabled"] is False
    assert payload["retriever"]["structured"]["multihop_rescue_enabled"] is False
