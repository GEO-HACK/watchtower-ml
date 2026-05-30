"""Owns anomaly scoring, confidence threshold optimization, and Isolation Forest escalation logic."""

import numpy as np
from sklearn.metrics import f1_score


def attack_score_from_proba(proba):
    """Return per-sample attack risk score from class probabilities.

    The score is the sum of all non-benign probabilities when probabilities are
    available. This is better than raw confidence because BENIGN can be highly
    confident without implying attack risk.
    """
    if proba is None:
        return None

    if proba.ndim != 2 or proba.shape[1] == 0:
        return None

    if proba.shape[1] == 1:
        return np.zeros(proba.shape[0], dtype=float)

    return np.sum(proba[:, 1:], axis=1)


def optimize_threshold(y_true, scores):
    """Find the score threshold that best separates attack vs benign."""
    if y_true is None or scores is None or len(scores) == 0:
        return None, None

    y_binary = (np.asarray(y_true) != 0).astype(int)
    scores = np.asarray(scores, dtype=float)

    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidates:
        preds_binary = (scores >= threshold).astype(int)
        score = f1_score(y_binary, preds_binary, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def apply_if_escalation(fused_preds, if_preds):
    final_preds = fused_preds.copy().astype(int)
    escalated = np.zeros(len(final_preds), dtype=bool)
    for i in range(len(final_preds)):
        if final_preds[i] == 0 and if_preds[i] == -1:
            final_preds[i] = -1
            escalated[i] = True
    return final_preds, escalated