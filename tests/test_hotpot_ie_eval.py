from pathlib import Path
import importlib.util


def _load_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "hotpotqa" / "eval_ie_curve.py"
    spec = importlib.util.spec_from_file_location("eval_ie_curve_mod", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[call-arg]
    return mod


def test_ie_samples_with_padding_behavior():
    mod = _load_eval_module()
    gold = {
        "q1": {"exact": {"Title A"}, "lowered": {"title a"}},
        "q2": {"exact": {"Title B"}, "lowered": {"title b"}},
    }
    ranked = {
        "q1": ["Title A", "Noise X"],
        "q2": ["Noise Y"],
    }
    samples = mod._compute_ie_samples(gold, ranked, [1, 2, 3])
    # q1: [1/1, 1/2, 1/3]
    # q2: [0/1, 0/2 (missing padded), 0/3 (missing padded)]
    assert samples[1] == [1.0, 0.0]
    assert samples[2] == [0.5, 0.0]
    assert samples[3] == [1.0 / 3.0, 0.0]


def test_bootstrap_ci_is_deterministic():
    mod = _load_eval_module()
    values = [0.0, 0.5, 1.0, 0.0]
    ci1 = mod._bootstrap_ci(values, samples=50, seed=42)
    ci2 = mod._bootstrap_ci(values, samples=50, seed=42)
    assert ci1 == ci2
