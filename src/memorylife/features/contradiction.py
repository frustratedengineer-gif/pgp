"""Contradiction detection (the "Contradiction (NLI)" box) -- off-the-shelf
NLI model (same checkpoint family as intent.py's zero-shot backbone), no
training. Checks each memory as the NLI hypothesis against its single most
similar EARLIER memory in the same conversation as the premise (causal, same
ordering rule as novelty.py -- comparing against later memories would leak
information a real system wouldn't have yet). A memory that directly
contradicts something said earlier is exactly the "update/invalidate" signal
this whole project is trying to predict, so this feature is expected to
correlate strongly with the Lifetime head and with the action head's
"update"/"contradiction" labels -- that's intentional, not leakage, since
the label used for training the action head (lifecycle_event) is derived
from the SYNTHETIC GENERATION process, not from this NLI model's output.
"""
from datetime import datetime

import numpy as np

from .base import FeatureExtractor

DEFAULT_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
NLI_LABELS = ("contradiction", "neutral", "entailment")


class ContradictionFeatures(FeatureExtractor):
    """3 features: [P(contradiction), P(neutral), P(entailment)] against the
    most similar earlier memory in the same conversation. A record with no
    earlier memories gets [0, 1, 0] (neutral -- nothing to contradict)."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        from transformers import pipeline
        self._pipe = pipeline("text-classification", model=model_name, top_k=None,
                               device=0 if device == "cuda" else -1)

    @property
    def dim(self) -> int:
        return len(NLI_LABELS)

    @property
    def name(self) -> str:
        return "contradiction"

    def extract(self, records: list[dict], embeddings: np.ndarray | None = None) -> np.ndarray:
        if embeddings is None:
            raise ValueError("ContradictionFeatures requires embeddings (finds the nearest prior memory)")

        by_conv: dict[str, list[int]] = {}
        for idx, r in enumerate(records):
            by_conv.setdefault(r["conversation_id"], []).append(idx)

        out = np.zeros((len(records), self.dim), dtype=np.float32)
        out[:, NLI_LABELS.index("neutral")] = 1.0  # default: nothing to compare against

        pairs = []  # (record_idx, premise_text)
        for conv_id, idxs in by_conv.items():
            idxs_sorted = sorted(idxs, key=lambda i: datetime.fromisoformat(records[i]["injected_at"]))
            seen_idx = []
            seen_emb = []
            for i in idxs_sorted:
                if seen_emb:
                    sims = embeddings[i] @ np.stack(seen_emb).T
                    nearest = seen_idx[int(sims.argmax())]
                    pairs.append((i, records[nearest]["text"][:500]))
                seen_idx.append(i)
                seen_emb.append(embeddings[i])

        if not pairs:
            return out

        batch_inputs = [{"text": premise, "text_pair": records[i]["text"][:500]} for i, premise in pairs]
        results = self._pipe(batch_inputs, batch_size=64)
        for (i, _premise), scores in zip(pairs, results):
            for item in scores:
                label = item["label"].lower()
                if label in NLI_LABELS:
                    out[i, NLI_LABELS.index(label)] = item["score"]
        return out
