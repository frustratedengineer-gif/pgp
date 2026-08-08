"""
Schema for a single MemoryLifeBench record (data/*_survival.jsonl).

Plain dataclasses on purpose -- no pydantic dependency was added to the
environment the Week 1-3 results were produced in (see requirements.txt),
so this module stays a documented, honest description of the JSONL shape
rather than a new runtime dependency nothing else exercises yet.
"""
from dataclasses import dataclass, field
from typing import Optional

VALID_SOURCES = ("synthetic", "longmemeval", "locomo")
VALID_LIFECYCLE_EVENTS = (
    "none", "update", "contradiction", "natural_expiry",
    "observed_usage", "no_usage_observed",
)
VALID_CENSOR_REASONS = (
    "observed_event", "censored_last_probe", "censored_conversation_end",
)


@dataclass
class Probe:
    probe_at: str  # ISO-8601 timestamp
    expected_answer_source: str  # "original" | "updated_or_null"


@dataclass
class MemoryRecord:
    memory_id: str
    source: str  # one of VALID_SOURCES
    conversation_id: str
    category: str
    injected_at: str  # ISO-8601: when the memory-bearing statement was made
    lifetime_is_exact: bool
    text: str
    lifecycle_event: str  # one of VALID_LIFECYCLE_EVENTS
    censored: bool
    invalidated_at: Optional[str] = None  # ISO-8601; None iff censored

    probes: list = field(default_factory=list)  # list[Probe]
    session_id: Optional[int] = None
    fact_slot: Optional[str] = None
    update_text: Optional[str] = None
    speaker: Optional[str] = None
    evidence_dia_id: Optional[str] = None

    # populated by data.event_labeling; absent in the raw input files
    duration_days: Optional[float] = None
    event_observed: Optional[int] = None
    censor_reason: Optional[str] = None  # one of VALID_CENSOR_REASONS

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        d = dict(d)
        d["probes"] = [Probe(**p) for p in d.get("probes", [])]
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate(self) -> None:
        assert self.source in VALID_SOURCES, f"bad source: {self.source}"
        assert self.lifecycle_event in VALID_LIFECYCLE_EVENTS, \
            f"bad lifecycle_event: {self.lifecycle_event}"
        assert self.censored == (self.invalidated_at is None), (
            "censored must be True iff invalidated_at is None "
            f"(memory_id={self.memory_id})"
        )
        if self.censor_reason is not None:
            assert self.censor_reason in VALID_CENSOR_REASONS
