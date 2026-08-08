"""Split-integrity checks and split-ID persistence (data/splits/*.txt)."""
from pathlib import Path


def conversation_ids(records: list[dict]) -> set[str]:
    return {r["conversation_id"] for r in records}


def assert_no_leakage(split_records: dict[str, list[dict]]) -> None:
    """split_records: {"train": [...], "val": [...], "test": [...]}.
    Raises AssertionError if any conversation_id appears in more than one split."""
    ids_by_split = {name: conversation_ids(recs) for name, recs in split_records.items()}
    names = list(ids_by_split)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = ids_by_split[names[i]] & ids_by_split[names[j]]
            assert not overlap, (
                f"conversation_id leakage between {names[i]} and {names[j]}: "
                f"{len(overlap)} shared conversations"
            )


def write_split_ids(records: list[dict], path: str | Path) -> None:
    """Commit memory_id -> split mapping so splits are reproducible without
    redistributing the underlying data."""
    ids = [r["memory_id"] for r in records]
    Path(path).write_text("\n".join(ids) + "\n", encoding="utf-8")


def read_split_ids(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()
