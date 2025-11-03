from typing import Callable, Iterable, Tuple

from .mirage import iter_docs_and_chunks as _mirage_iter

Adapter = Callable[[str, int, int], Iterable[Tuple[dict, dict]]]

ADAPTERS: dict[str, Adapter] = {
    "mirage": _mirage_iter,
}


def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset adapter: {name}") from exc
