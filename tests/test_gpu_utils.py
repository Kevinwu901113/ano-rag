import importlib.util
import os
import sys
import types

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GPU_UTILS_PATH = os.path.join(ROOT_DIR, "utils", "gpu_utils.py")


def load_gpu_utils(mock_cudf=False):
    # Provide minimal stub modules so gpu_utils can be imported without heavy dependencies
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda *_: types.SimpleNamespace(total_memory=0),
            memory_allocated=lambda: 0,
            memory_reserved=lambda: 0,
        )
    )
    fake_numpy = types.SimpleNamespace(ndarray=list)
    fake_loguru = types.SimpleNamespace(
        logger=types.SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None, info=lambda *a, **k: None, error=lambda *a, **k: None)
    )

    sys.modules.setdefault("torch", fake_torch)
    sys.modules.setdefault("numpy", fake_numpy)
    sys.modules.setdefault("loguru", fake_loguru)

    if mock_cudf:
        class FakeIndexer:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, slc):
                return FakeDF(self._data[slc])

        class FakeDF:
            def __init__(self, data):
                self.data = list(data)
                self._iloc = FakeIndexer(self.data)

            def __len__(self):
                return len(self.data)

            @property
            def iloc(self):
                return self._iloc

            def to_pandas(self):
                return types.SimpleNamespace(values=self.data)

        fake_df_cls = FakeDF
        fake_cudf = types.SimpleNamespace(DataFrame=fake_df_cls)
        sys.modules.setdefault("cudf", fake_cudf)
        sys.modules.setdefault("cuml", types.ModuleType("cuml"))
        sys.modules.setdefault("cugraph", types.ModuleType("cugraph"))
    spec = importlib.util.spec_from_file_location("utils.gpu_utils", GPU_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_process_gpu_list():
    gpu_utils = load_gpu_utils()
    GPUUtils = gpu_utils.GPUUtils
    result = GPUUtils.batch_process_gpu([1, 2, 3, 4], 2, lambda b: [x * 2 for x in b], use_gpu=False)
    assert result == [2, 4, 6, 8]


def test_batch_process_gpu_cudf(monkeypatch):
    gpu_utils = load_gpu_utils(mock_cudf=True)
    GPUUtils = gpu_utils.GPUUtils
    monkeypatch.setattr(gpu_utils, "CUDF_AVAILABLE", True, raising=False)
    df = gpu_utils.cudf.DataFrame([1, 2, 3, 4])
    result = GPUUtils.batch_process_gpu(df, 2, lambda b: [x * 2 for x in b.data], use_gpu=True)
    assert result == [2, 4, 6, 8]

