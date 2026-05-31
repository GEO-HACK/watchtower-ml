import logging
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from api.config import settings
from diagnostics.schema_checker import inspect_model_file
from inference.fusion import combine_predictions, combine_scores
from inference.model_loader import get_class_names, load_maybe_dict_model, warmup_models
from inference.predictor import prepare_input
from preprocessing.preprocessing_pipeline1 import preprocess_for_inference
from utils.latency_tracker import LatencyTracker


logger = logging.getLogger(__name__)


class MLInferenceService:
    """ML inference layer that reuses the existing terminal detection pipeline."""

    def __init__(self) -> None:
        self.model1 = None
        self.model2 = None
        self.iso_forest = None
        self.class_names: List[str] = []
        self.model1_info: Dict[str, Any] = {}
        self.model2_info: Dict[str, Any] = {}
        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        self.model1, _ = load_maybe_dict_model(settings.model1_path)
        self.model2, _ = load_maybe_dict_model(settings.model2_path)
        self.iso_forest = joblib.load(settings.model3_path)
        self.class_names = get_class_names(self.model1)
        self.model1_info = inspect_model_file(settings.model1_path)
        self.model2_info = inspect_model_file(settings.model2_path)

        n_features = (
            self.model1_info.get("n_features_in_")
            or self.model2_info.get("n_features_in_")
            or getattr(self.iso_forest, "n_features_in_", None)
            or settings.expected_feature_count
        )
        warmup_models(self.model1, self.model2, self.iso_forest, n_features)
        logger.info("ML inference runtime initialized")

    def _class_name_for_prediction(self, prediction: int) -> str:
        if prediction == -1:
            return "ANOMALY"
        if 0 <= int(prediction) < len(self.class_names):
            return str(self.class_names[int(prediction)])
        return str(prediction)

    @staticmethod
    def _label_for_prediction(prediction: int) -> str:
        return "Normal" if int(prediction) == 0 else "Attack"

    def predict_from_dataframe(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        latency_tracker = LatencyTracker()

        model_input = prepare_input(df)
        latency_tracker.start("preprocessing")
        X = preprocess_for_inference(
            model_input,
            model1_info=self.model1_info,
            model2_info=self.model2_info,
            pipeline_path=settings.pipeline_path,
            expected_feature_count=settings.expected_feature_count,
        )
        latency_tracker.stop("preprocessing")

        if X is None or getattr(X, "size", 0) == 0:
            raise ValueError("No feature data available for inference")

        latency_tracker.start("rf_inference")
        proba1 = self.model1.predict_proba(X)
        preds1 = np.asarray(self.model1.classes_)[np.argmax(proba1, axis=1)]
        latency_tracker.stop("rf_inference")

        latency_tracker.start("xgb_inference")
        proba2 = self.model2.predict_proba(X)
        preds2 = np.asarray(self.model2.classes_)[np.argmax(proba2, axis=1)]
        latency_tracker.stop("xgb_inference")

        latency_tracker.start("if_inference")
        if_preds = self.iso_forest.predict(X)
        if_scores = self.iso_forest.score_samples(X)
        latency_tracker.stop("if_inference")

        if_score_min = if_scores.min()
        if_score_max = if_scores.max()
        if_attack_scores = 1.0 - ((if_scores - if_score_min) / (if_score_max - if_score_min + 1e-9))

        fused_results = {
            s: combine_predictions(preds1, preds2, proba1, proba2, strategy=s)
            for s in ["majority", "or", "confidence_weighted", "unanimous_or_majority"]
        }
        fused_scores = {
            s: combine_scores(preds1, preds2, proba1, proba2, strategy=s)
            for s in ["majority", "or", "confidence_weighted", "unanimous_or_majority"]
        }

        final_preds = fused_results["majority"].copy().astype(int)
        escalated = np.zeros(len(final_preds), dtype=bool)
        for idx in range(len(final_preds)):
            rf_xgb_said_benign = final_preds[idx] == 0
            if_said_anomaly = if_preds[idx] == -1
            if rf_xgb_said_benign and if_said_anomaly:
                final_preds[idx] = -1
                escalated[idx] = True

        results: List[Dict[str, Any]] = []
        for idx, final_pred in enumerate(final_preds):
            final_name = self._class_name_for_prediction(int(final_pred))
            attack_type = None if int(final_pred) == 0 else final_name
            confidence = float(if_attack_scores[idx]) if escalated[idx] else float(fused_scores["majority"][idx])

            results.append(
                {
                    "label": self._label_for_prediction(int(final_pred)),
                    "attack_type": attack_type,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )

        return results, latency_tracker.summary()
