"""
Metrics beyond C-index: time-dependent (cumulative/dynamic) AUC, Brier
score, and Integrated Brier Score (IBS). Uses scikit-survival's vetted IPCW
implementations (sksurv.metrics) rather than hand-rolling the censoring
adjustment, a common source of subtle bugs in survival analysis.

Time-dependent AUC only needs a risk score (this repo's usual convention:
higher score == longer predicted survival, so risk = -score) and applies
uniformly to every method, trained or not.

Brier score / IBS need a full survival probability curve S(t|x), which only
our_model actually produces (from the fitted Cox baseline hazards). The
scalar "predicted TTL in days" baselines (heuristic, bucket classifier,
every LLM-prompted method) don't have one -- they're scored under Brier's
proper scoring rule using a degenerate step-function curve
(S(t)=1 for t < predicted_days, else 0), the standard way to Brier-score a
point forecast. This is noted wherever used so it doesn't read as if those
methods produce calibrated probabilities -- they don't.
"""
import numpy as np
from sksurv.metrics import brier_score, cumulative_dynamic_auc, integrated_brier_score


def _structured(durations, events):
    return np.array(
        [(bool(e), float(d)) for e, d in zip(events, durations)],
        dtype=[("event", bool), ("time", float)],
    )


def eval_times_for(durations, n_times: int = 15) -> np.ndarray:
    """sksurv requires eval times strictly within the observed follow-up
    range and complains near the boundary -- pull in from the raw
    percentiles rather than using the full min/max."""
    lo, hi = np.percentile(durations, [5, 95])
    lo = max(lo, float(np.min(durations)) + 1e-3)
    hi = min(hi, float(np.max(durations)) - 1e-3)
    return np.linspace(lo, hi, n_times)


def time_dependent_auc(train_durations, train_events, durations, events, scores, eval_times) -> dict:
    """scores: higher == predicted to survive longer (this repo's convention)."""
    y_train = _structured(train_durations, train_events)
    y_eval = _structured(durations, events)
    risk = -np.asarray(scores, dtype=float)
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_eval, risk, eval_times)
    return {"times": eval_times.tolist(), "auc": auc.tolist(), "mean_auc": float(mean_auc)}


def step_function_surv_probs(predicted_days, eval_times: np.ndarray) -> np.ndarray:
    """Degenerate S(t|x) for a scalar point-forecast baseline: 1 while
    t < predicted_days, 0 after. Shape (n_records, n_times)."""
    predicted_days = np.asarray(predicted_days, dtype=float).reshape(-1, 1)
    return (eval_times.reshape(1, -1) < predicted_days).astype(float)


def cox_surv_probs(model, x_eval, eval_times: np.ndarray) -> np.ndarray:
    """Real S(t|x) from the fitted CoxPH model's baseline hazards, resampled
    onto eval_times. model must already have compute_baseline_hazards() run
    on the training split (checkpoints don't persist it -- see
    scripts/compute_richer_metrics.py)."""
    surv_df = model.predict_surv_df(x_eval.astype("float32"))
    if 0.0 not in surv_df.index:
        surv_df.loc[0.0] = 1.0
    surv_df = surv_df.sort_index()
    combined = surv_df.index.union(eval_times)
    surv_full = surv_df.reindex(combined).ffill()
    return surv_full.loc[eval_times].to_numpy().T  # (n_records, n_times)


def brier_and_ibs(train_durations, train_events, durations, events, surv_probs, eval_times) -> dict:
    """surv_probs: (n_records, n_times) array of S(t|x) at eval_times."""
    y_train = _structured(train_durations, train_events)
    y_eval = _structured(durations, events)
    times, scores = brier_score(y_train, y_eval, surv_probs, eval_times)
    ibs = integrated_brier_score(y_train, y_eval, surv_probs, eval_times)
    return {"times": times.tolist(), "brier": scores.tolist(), "ibs": float(ibs)}
