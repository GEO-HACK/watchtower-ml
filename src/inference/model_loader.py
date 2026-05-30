"""Owns loading serialized models from disk, runtime parameter configuration, class name resolution, and model warm-up."""

import logging

import joblib
import numpy as np


logger = logging.getLogger(__name__)


DEFAULT_CLASS_NAMES = [
    'BENIGN',
    'Bot',
    'DDoS',
    'DoS GoldenEye',
    'DoS Hulk',
    'DoS Slowhttptest',
    'DoS slowloris',
    'FTP-Patator',
    'PortScan',
    'SSH-Patator',
]
def load_maybe_dict_model(path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
    else:
        model = obj

    if hasattr(model, 'n_jobs'):
        try:
            model.n_jobs = -1
        except Exception:
            pass

    if hasattr(model, 'verbose'):
        try:
            model.verbose = 0
        except Exception:
            pass

    model_type_name = type(model).__name__.lower()
    model_module_name = type(model).__module__.lower()
    looks_like_xgb = 'xgb' in model_type_name or 'xgboost' in model_module_name

    if looks_like_xgb and hasattr(model, 'set_params'):
        try:
            model.set_params(nthread=-1, predictor='cpu_predictor')
            try:
                model.set_params(predictor='gpu_predictor')
                logger.info('XGBoost GPU predictor enabled')
            except Exception as e:
                logger.info('XGBoost GPU predictor unavailable; using CPU predictor (%s)', e)
                model.set_params(predictor='cpu_predictor')
        except Exception as e:
            logger.debug('Could not configure XGBoost runtime parameters: %s', e)

    return model, obj


def get_class_names(model, fallback=None):
    classes = getattr(model, 'classes_', None)
    fallback = fallback or DEFAULT_CLASS_NAMES
    if classes is None:
        return fallback

    class_list = list(classes)
    if all(isinstance(cls, (int, np.integer)) for cls in class_list):
        names = []
        for cls in class_list:
            cls_idx = int(cls)
            if 0 <= cls_idx < len(fallback):
                names.append(fallback[cls_idx])
            else:
                names.append(str(cls_idx))
        return names

    return [str(cls) for cls in class_list]


def warmup_models(rf_model, xgb_model, iso_model, n_features):
    """Perform a single dummy inference on each model to warm up threads/caches."""
    try:
        n = int(n_features) if n_features is not None else 70
    except Exception:
        n = 70

    dummy = np.zeros((1, n), dtype=np.float32)
    try:
        if rf_model is not None and hasattr(rf_model, 'predict_proba'):
            rf_model.predict_proba(dummy)
    except Exception as e:
        logger.debug('RF warmup failed: %s', e)

    try:
        if xgb_model is not None and hasattr(xgb_model, 'predict_proba'):
            xgb_model.predict_proba(dummy)
    except Exception as e:
        logger.debug('XGB warmup failed: %s', e)

    try:
        if iso_model is not None:
            if hasattr(iso_model, 'decision_function'):
                iso_model.decision_function(dummy)
            else:
                iso_model.predict(dummy)
    except Exception as e:
        logger.debug('IF warmup failed: %s', e)