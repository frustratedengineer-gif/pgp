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
import numpy as np

from .base import FeatureExtractor
from .causal import nearest_prior_in_conversation

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

        out = np.zeros((len(records), self.dim), dtype=np.float32)
        out[:, NLI_LABELS.index("neutral")] = 1.0  # default: nothing to compare against

        prior = nearest_prior_in_conversation(records, embeddings)
        pairs = [(i, records[prior_i]["text"][:500]) for i, prior_i in prior.items() if prior_i is not None]
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
