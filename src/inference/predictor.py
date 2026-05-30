"""Owns input normalization, single-model inference calls, and per-sample confidence score extraction."""

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def prepare_input(raw_input) -> np.ndarray:
    """
    Accepts: dict, list, pd.Series, pd.DataFrame, or np.ndarray.
    Returns: np.float32 numpy array with shape (1, n_features) for single flow,
             or (n, n_features) for batch.
    """
    if isinstance(raw_input, np.ndarray):
        return raw_input.astype(np.float32)
    if isinstance(raw_input, pd.DataFrame):
        # Don't force DataFrame -> numpy here; let the preprocessing pipeline
        # handle DataFrame extraction and type conversion (it knows which
        # columns are numeric). Returning the DataFrame preserves string
        # columns needed by feature alignment.
        return raw_input
    if isinstance(raw_input, (dict, pd.Series)):
        return np.array(list(raw_input.values()), dtype=np.float32).reshape(1, -1)
    return np.array(raw_input, dtype=np.float32).reshape(1, -1)


def predict_with_model(model, X):
    preds = model.predict(X)          # hard predictions — use model directly
    proba = None
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)  # only for fusion scoring
    except Exception as e:
        logger.warning('predict_proba failed: %s', e)
    return preds, proba


def prediction_confidence(preds, proba):
    if proba is None or len(preds) == 0:
        return np.ones(len(preds), dtype=float)

    pred_indices = np.argmax(proba, axis=1)
    return proba[np.arange(len(preds)), pred_indices]


def class_name_for_prediction(prediction, class_names):
    try:
        prediction_index = int(prediction)
        if 0 <= prediction_index < len(class_names):
            return class_names[prediction_index]
    except Exception:
        pass
    return str(prediction)