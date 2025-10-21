import importlib.util
import sys
import types
from pathlib import Path


def load_main_mirage():
    # Inject a lightweight MIRAGE.utils stub so the CLI module can be imported in tests
    if "MIRAGE.utils" not in sys.modules:
        mirage_pkg = types.ModuleType("MIRAGE")
        utils_mod = types.ModuleType("MIRAGE.utils")

        def _load_json(path):  # pragma: no cover - simple stub for import-time dependency
            return []

        def _convert_doc_pool(doc_pool):  # pragma: no cover - simple stub
            return doc_pool

        utils_mod.load_json = _load_json
        utils_mod.convert_doc_pool = _convert_doc_pool

        sys.modules["MIRAGE"] = mirage_pkg
        sys.modules["MIRAGE.utils"] = utils_mod

    module_path = Path(__file__).resolve().parents[1] / "main_mirage.py"
    spec = importlib.util.spec_from_file_location("main_mirage", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to load main_mirage module"
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


MAIN_MIRAGE = load_main_mirage()


def parse_args(argv):
    parser = MAIN_MIRAGE.build_parser()
    # Ensure parser doesn't exit on parse errors during tests
    parser.exit = lambda status=0, message=None: (_ for _ in ()).throw(AssertionError(message))
    return parser.parse_args(argv)


def test_embed_model_aliases():
    args_dash = parse_args(['--embed-model', 'dash-model'])
    assert args_dash.embed_model == 'dash-model'

    args_underscore = parse_args(['--embed_model', 'underscore-model'])
    assert args_underscore.embed_model == 'underscore-model'


def test_rebuild_index_aliases():
    args_dash = parse_args(['--rebuild-index'])
    assert args_dash.rebuild_index is True

    args_underscore = parse_args(['--rebuild_index'])
    assert args_underscore.rebuild_index is True
