"""Custom preprocessing transformers for the Watchtower IDS pipeline.

Why this module exists:
- sklearn/joblib serializes custom estimators by module path.
- If a transformer is defined in a notebook cell, its module is usually
  __main__, which only exists in that interactive session.
- Loading that pickle later from a normal script fails because the original
  __main__.FeatureAlignTransformer cannot be resolved.

Keeping custom transformers in an importable Python module makes the saved
pipeline portable across notebooks, scripts, and inference jobs.
"""

from __future__ import annotations

from typing import Sequence

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


def align_feature_columns(
    X: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Align an input matrix to the expected training feature order."""
    expected_features = list(feature_names)

    if isinstance(X, pd.DataFrame):
        aligned = X.copy()
    else:
        aligned = pd.DataFrame(X, columns=expected_features)

    missing = [column for column in expected_features if column not in aligned.columns]
    if missing:
        logger.warning("FeatureAlignTransformer added %d missing columns with NaN", len(missing))
        for column in missing:
            aligned[column] = np.nan

    extra = [column for column in aligned.columns if column not in expected_features]
    if extra:
        logger.info("FeatureAlignTransformer dropped %d extra columns", len(extra))

    return aligned[expected_features]


class FeatureAlignTransformer(BaseEstimator, TransformerMixin):
    """Align inference data to the exact feature order used during training."""

    def __init__(self, feature_names: Sequence[str]):
        self.feature_names = list(feature_names)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            if hasattr(X, "shape") and len(X.shape) == 2 and X.shape[1] == len(self.feature_names):
                X = pd.DataFrame(X, columns=self.feature_names)
            else:
                raise ValueError(
                    "FeatureAlignTransformer expects a pandas.DataFrame or a 2D array "
                    "with the exact training feature count."
                )

        return align_feature_columns(X, self.feature_names)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names, dtype=object)


def build_preprocessing_pipeline(feature_names: Sequence[str]) -> Pipeline:
    """Create the canonical preprocessing pipeline used by training and inference."""
    return Pipeline(
        steps=[
            ("feature_aligner", FeatureAlignTransformer(feature_names)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
