"""Owns all multi-model fusion strategies for combining RF and XGB predictions and attack scores."""

import numpy as np


def combine_predictions(preds1, preds2, proba1=None, proba2=None, strategy="majority"):
    """Fusion layer: combine predictions from two models."""
    if strategy == "or":
        combined = []
        for a, b in zip(preds1, preds2):
            if a != 0 and b != 0:
                combined.append(a)
            elif a != 0:
                combined.append(a)
            elif b != 0:
                combined.append(b)
            else:
                combined.append(0)
        return np.array(combined)

    if strategy == "avg_proba" and proba1 is not None and proba2 is not None:
        avg = (proba1 + proba2) / 2.0
        return np.argmax(avg, axis=1)

    if strategy == "confidence_weighted" and proba1 is not None and proba2 is not None:
        c1 = np.max(proba1, axis=1)
        c2 = np.max(proba2, axis=1)
        combined = []
        for i, (a, b) in enumerate(zip(preds1, preds2)):
            if c1[i] > c2[i]:
                combined.append(a)
            else:
                combined.append(b)
        return np.array(combined)

    if strategy == "unanimous_or_majority":
        combined = []
        for a, b in zip(preds1, preds2):
            if a == b:
                combined.append(a)
            elif a != 0 or b != 0:
                combined.append(max(a, b) if a != 0 else b if b != 0 else 0)
            else:
                combined.append(a)
        return np.array(combined)

    # default majority
    combined = []
    for a, b in zip(preds1, preds2):
        if a == b:
            combined.append(a)
        else:
            combined.append(a)
    return np.array(combined)


def combine_scores(preds1, preds2, proba1=None, proba2=None, strategy="majority"):
    """Return an attack score for each merged prediction."""
    if proba1 is None:
        score1 = np.zeros(len(preds1), dtype=float)
    elif proba1.ndim != 2 or proba1.shape[1] == 0:
        score1 = np.zeros(len(preds1), dtype=float)
    elif proba1.shape[1] == 1:
        score1 = np.zeros(proba1.shape[0], dtype=float)
    else:
        score1 = np.sum(proba1[:, 1:], axis=1)

    if proba2 is None:
        score2 = np.zeros(len(preds2), dtype=float)
    elif proba2.ndim != 2 or proba2.shape[1] == 0:
        score2 = np.zeros(len(preds2), dtype=float)
    elif proba2.shape[1] == 1:
        score2 = np.zeros(proba2.shape[0], dtype=float)
    else:
        score2 = np.sum(proba2[:, 1:], axis=1)

    if strategy == 'avg_proba' and proba1 is not None and proba2 is not None:
        avg = (proba1 + proba2) / 2.0
        return np.sum(avg[:, 1:], axis=1) if avg.shape[1] > 1 else np.zeros(avg.shape[0], dtype=float)

    if strategy == 'confidence_weighted':
        return np.maximum(score1, score2)

    merged_scores = []
    for a, b, s1, s2 in zip(preds1, preds2, score1, score2):
        if a == b:
            merged_scores.append((s1 + s2) / 2.0)
        elif a != 0 and b == 0:
            merged_scores.append(s1)
        elif b != 0 and a == 0:
            merged_scores.append(s2)
        else:
            merged_scores.append(max(s1, s2))

    return np.asarray(merged_scores, dtype=float)