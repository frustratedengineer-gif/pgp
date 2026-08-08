"""
Baseline 2: day / week / permanent classifier.

A 3-class logistic regression on the frozen BGE embeddings, predicting a
coarse lifetime bucket. Supervision comes only from records with an
*observed* event (censored=False), since only those have a known true
duration. Predictions on all records are mapped back to a numeric
TTL-in-days proxy so they're comparable to the other methods under the same
concordance-index evaluation.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from memorylife.data.datasets import load_split
from memorylife.encoders.cache import ensure_embeddings

DAY_MAX = 2.0
WEEK_MAX = 14.0
BUCKET_TTL = {"day": 1.0, "week": 7.0, "permanent": 365.0}


def bucket_of(duration_days: float) -> str:
    if duration_days <= DAY_MAX:
        return "day"
    if duration_days <= WEEK_MAX:
        return "week"
    return "permanent"


def run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_embeddings(args.data_dir, args.emb_dir, list(set(args.splits) | {"train"}),
                       device=getattr(args, "device", "cuda"))

    train = load_split(args.data_dir, args.emb_dir, "train")
    observed = train["events"] == 1
    x_sup = train["embeddings"][observed]
    y_sup = np.array([bucket_of(d) for d in train["durations"][observed]])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_sup, y_sup)

    for split_name in args.splits:
        split = train if split_name == "train" else load_split(args.data_dir, args.emb_dir, split_name)
        pred_bucket = clf.predict(split["embeddings"])
        scores = {mid: BUCKET_TTL[b] for mid, b in zip(split["ids"], pred_bucket)}
        with open(out_dir / f"bucket_classifier_{split_name}.json", "w") as f:
            json.dump(scores, f)
        print(f"{split_name}: wrote {len(scores)} scores -> bucket_classifier_{split_name}.json")
