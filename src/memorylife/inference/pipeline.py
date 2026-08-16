"""
End-to-end inference pipeline (statement -> memory -> query -> answer),
the "LLM -> grounded answer" box tying the rest of Week 5 together.

Design note on which model predicts what (see also memory/memory_object.py):
  - predicted_ttl_days comes from the Week-3/4 lone survival head
    (models/checkpoint.load_survival_model), via pycox's own
    predict_surv_df/baseline-hazard machinery -- this is the model already
    validated by C-index/significance/Brier-IBS across Weeks 3-4, so the
    concrete "when do we expire this" number uses the most rigorously
    checked source available.
  - action and utility_prob come from the Week-5 joint model
    (models/checkpoint.load_joint_model) -- there is no standalone
    action/utility model to load; they only exist as heads inside the
    joint model.
  - importance comes from heads/importance.py's heuristic.
This is a deliberate, documented two-model split, not an oversight -- see
results/tables/week5_joint_model_results.md for why the joint model's own
survival head is reported separately (as an ablation) rather than swapped
in here.

build_memory_objects() ingests already-labeled MemoryLifeBench records
(text is already extracted) -- it does NOT call the memory_extraction
prompt. That prompt exists for a future live-deployment path (raw dialogue
turns -> extracted memories), documented but not exercised by this repo's
data, since MemoryLifeBench's records are themselves the ground truth for
Weeks 1-4 and are already single, self-contained statements.
"""
from pathlib import Path

import numpy as np
import torch

from ..heads.action import ACTION_LABELS
from ..heads.importance import importance_score
from ..memory.memory_object import MemoryObject
from ..memory.store.base import MemoryStore
from ..retrieval.retriever import Retriever
from .llm_client import DEFAULT_MODEL, chat_completion

PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_SURVIVAL_TTL_DAYS = 3650.0  # cap for records whose survival curve never crosses 0.5 (predicted near-permanent)


def quantile_ttl_days(surv_df, quantile: float = 0.5) -> np.ndarray:
    """surv_df: pycox's predict_surv_df output (index=time, columns=records).
    TTL per column: the last time point where S(t) >= quantile. quantile=0.5
    (the default, and the only value used through Week 6's headline runs)
    is the MEDIAN survival time -- and, used as a hard deterministic
    eviction cutoff, is inherently a coin-flip threshold: by definition of
    "median," roughly half of records whose curve is correctly calibrated
    are still "alive" past it. C-index (this model's validated metric,
    Weeks 3-5) is rank-only and invariant to monotonic rescaling of the
    risk score -- it never validates that any one quantile is a safe
    absolute cutoff. Passing a lower quantile (e.g. 0.2 or 0.1, i.e. the
    80th/90th percentile survival time) shifts the cutoff later, trading
    storage for retention; see scripts/diagnose_eviction_evidence.py and
    results/tables/week6_ttl_quantile_sweep.md for the measured effect.
    Records whose curve never drops below `quantile` (predicted durable)
    get MAX_SURVIVAL_TTL_DAYS rather than an unbounded/undefined value."""
    times = surv_df.index.to_numpy()
    values = surv_df.to_numpy()  # (n_times, n_records)
    out = np.full(values.shape[1], MAX_SURVIVAL_TTL_DAYS, dtype=np.float32)
    for col in range(values.shape[1]):
        below = np.where(values[:, col] < quantile)[0]
        if len(below) > 0:
            idx = max(below[0] - 1, 0)
            out[col] = float(times[idx])
    return out


def build_memory_objects(records: list[dict], embeddings: np.ndarray, features: np.ndarray,
                          feature_slices: dict, survival_model, joint_model, device: str = "cpu",
                          ttl_quantile: float = 0.5) -> list[MemoryObject]:
    importances = importance_score(features, feature_slices)

    surv_df = survival_model.predict_surv_df(embeddings.astype("float32"))
    ttl_days = quantile_ttl_days(surv_df, ttl_quantile)

    joint_model.eval()
    with torch.no_grad():
        emb_t = torch.tensor(embeddings, dtype=torch.float32, device=device)
        feat_t = torch.tensor(features, dtype=torch.float32, device=device)
        out = joint_model(emb_t, feat_t)
        action_idx = out["action_logits"].argmax(dim=-1).cpu().numpy()
        utility_probs = torch.sigmoid(out["utility_logit"]).cpu().numpy()

    objects = []
    for i, r in enumerate(records):
        objects.append(MemoryObject(
            memory_id=r["memory_id"], text=r["text"], embedding=embeddings[i],
            importance=float(importances[i]), predicted_ttl_days=float(ttl_days[i]),
            action=ACTION_LABELS[action_idx[i]], utility_prob=float(utility_probs[i]),
            conversation_id=r["conversation_id"], source=r["source"], category=r["category"],
            created_at=r["injected_at"], provenance={"lifecycle_event": r.get("lifecycle_event")},
        ))
    return objects


def format_retrieved_block(results: list[tuple[MemoryObject, float]]) -> str:
    """Includes each memory's created_at date -- a large fraction of
    real QA questions (e.g. LoCoMo's temporal-reasoning category, see
    scripts/run_downstream_qa_eval.py) ask "when did X happen", which is
    unanswerable from text alone even with perfect retrieval if the date
    isn't surfaced to the LLM."""
    lines = []
    for obj, score in results:
        date_str = obj.created_at.split("T")[0]  # ISO date only, not the full timestamp
        lines.append(f'- [{date_str}] "{obj.text}" (relevance score {score:.2f}, '
                      f'predicted utility {obj.utility_prob:.2f})')
    return "\n".join(lines) if lines else "(no relevant memories found)"


class GroundedQAPipeline:
    def __init__(self, retriever: Retriever, llm_model: str = DEFAULT_MODEL):
        self.retriever = retriever
        self.llm_model = llm_model
        self._prompt_template = (PROMPTS_DIR / "qa_grounded.txt").read_text()

    def answer(self, query: str, k: int = 5) -> dict:
        results = self.retriever.retrieve(query, k=k)
        prompt = self._prompt_template.format(
            retrieved_memories_block=format_retrieved_block(results), query=query,
        )
        text, usage = chat_completion([{"role": "user", "content": prompt}], model=self.llm_model)
        return {
            "query": query,
            "answer": text,
            "retrieved_memories": [{"text": obj.text, "score": score, "memory_id": obj.memory_id}
                                    for obj, score in results],
            "usage": usage,
        }
