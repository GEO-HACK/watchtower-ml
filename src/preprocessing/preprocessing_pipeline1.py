import os
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class FeatureAlignTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        for col in [c for c in self.feature_names if c not in X.columns]:
            X[col] = np.nan
        return X[self.feature_names]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=10.0):
        self.factor = factor

    def fit(self, X, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return input_features


try:
    import sys

    _main_module = sys.modules.get('__main__')
    if _main_module is not None:
        _main_module.FeatureAlignTransformer = FeatureAlignTransformer
        _main_module.OutlierClipper = OutlierClipper
except Exception:
    pass


def _load_joblib_artifact(path):
    return joblib.load(path)


EXPECTED_FEATURE_COUNT = 70

# Mapping from PCAP extractor snake_case names to CICIDS Title Case names
PCAP_TO_CICIDS = {
    'src_port':                    'Source Port',
    'dst_port':                    'Destination Port',
    'protocol':                    'Protocol',
    'flow_duration':               'Flow Duration',
    'total_fwd_packets':           'Total Fwd Packets',
    'total_bwd_packets':           'Total Backward Packets',
    'total_fwd_bytes':             'Total Length of Fwd Packets',
    'total_bwd_bytes':             'Total Length of Bwd Packets',
    'fwd_packet_length_max':       'Fwd Packet Length Max',
    'fwd_packet_length_min':       'Fwd Packet Length Min',
    'fwd_packet_length_mean':      'Fwd Packet Length Mean',
    'fwd_packet_length_std':       'Fwd Packet Length Std',
    'bwd_packet_length_max':       'Bwd Packet Length Max',
    'bwd_packet_length_min':       'Bwd Packet Length Min',
    'bwd_packet_length_mean':      'Bwd Packet Length Mean',
    'bwd_packet_length_std':       'Bwd Packet Length Std',
    'flow_bytes_per_s':            'Flow Bytes/s',
    'flow_packets_per_s':          'Flow Packets/s',
    'flow_iat_mean':               'Flow IAT Mean',
    'flow_iat_std':                'Flow IAT Std',
    'flow_iat_max':                'Flow IAT Max',
    'flow_iat_min':                'Flow IAT Min',
    'fwd_iat_total':               'Fwd IAT Total',
    'fwd_iat_mean':                'Fwd IAT Mean',
    'fwd_iat_std':                 'Fwd IAT Std',
    'fwd_iat_max':                 'Fwd IAT Max',
    'fwd_iat_min':                 'Fwd IAT Min',
    'bwd_iat_total':               'Bwd IAT Total',
    'bwd_iat_mean':                'Bwd IAT Mean',
    'bwd_iat_std':                 'Bwd IAT Std',
    'bwd_iat_max':                 'Bwd IAT Max',
    'bwd_iat_min':                 'Bwd IAT Min',
    'fwd_psh_flags':               'Fwd PSH Flags',
    'fwd_header_length':           'Fwd Header Length',
    'bwd_header_length':           'Bwd Header Length',
    'fwd_packets_per_s':           'Fwd Packets/s',
    'bwd_packets_per_s':           'Bwd Packets/s',
    'min_packet_length':           'Min Packet Length',
    'max_packet_length':           'Max Packet Length',
    'packet_length_mean':          'Packet Length Mean',
    'packet_length_std':           'Packet Length Std',
    'packet_length_variance':      'Packet Length Variance',
    'fin_flag_count':              'FIN Flag Count',
    'syn_flag_count':              'SYN Flag Count',
    'rst_flag_count':              'RST Flag Count',
    'psh_flag_count':              'PSH Flag Count',
    'ack_flag_count':              'ACK Flag Count',
    'urg_flag_count':              'URG Flag Count',
    'ece_flag_count':              'ECE Flag Count',
    'down_up_ratio':               'Down/Up Ratio',
    'average_packet_size':         'Average Packet Size',
    'avg_fwd_segment_size':        'Avg Fwd Segment Size',
    'avg_bwd_segment_size':        'Avg Bwd Segment Size',
    'subflow_fwd_packets':         'Subflow Fwd Packets',
    'subflow_fwd_bytes':           'Subflow Fwd Bytes',
    'subflow_bwd_packets':         'Subflow Bwd Packets',
    'subflow_bwd_bytes':           'Subflow Bwd Bytes',
    'init_win_bytes_forward':      'Init Win bytes forward',
    'init_win_bytes_backward':     'Init Win bytes backward',
    'min_seg_size_forward':        'min seg size forward',
    'active_mean':                 'Active Mean',
    'active_std':                  'Active Std',
    'active_max':                  'Active Max',
    'active_min':                  'Active Min',
    'idle_mean':                   'Idle Mean',
    'idle_std':                    'Idle Std',
    'idle_max':                    'Idle Max',
    'idle_min':                    'Idle Min',
}


_PIPELINE_CANDIDATES = [
    os.path.join('src', 'models', 'preprocessing_pipeline.pkl'),
    os.path.join('src', 'preprocessing', 'preprocessing_pipeline.pkl'),
    os.path.join('src', 'preprocessing', 'preprocessing_pipeline1.pkl'),
]


def clean_flow_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Deterministic cleaning applied before pipeline.transform().

    - Operates on a copy (does not mutate input).
    - Strips leading/trailing whitespace from column names.
    - Replaces +/-inf with NaN.
    """
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f'clean_flow_features expects DataFrame, got {type(df)!r}')

    df_clean = df.copy()
    df_clean.columns = [str(column).strip() for column in df_clean.columns]

    try:
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    except Exception:
        vals = df_clean.to_numpy(dtype=object, copy=False)
        vals[np.isinf(vals.astype(float))] = np.nan
        df_clean = pd.DataFrame(vals, columns=df_clean.columns, index=df_clean.index)

    return df_clean


def _resolve_pipeline_path(explicit_path=None):
    candidate_errors = []

    if explicit_path is not None:
        if os.path.exists(explicit_path):
            try:
                pipeline = _load_joblib_artifact(explicit_path)
                _validate_pipeline_structure(pipeline, EXPECTED_FEATURE_COUNT)
                return explicit_path
            except Exception as exc:
                candidate_errors.append(f'{explicit_path}: {exc}')
        logger.warning('Configured preprocessing pipeline not found: %s', explicit_path)

    for candidate in _PIPELINE_CANDIDATES:
        if os.path.exists(candidate):
            try:
                pipeline = _load_joblib_artifact(candidate)
                _validate_pipeline_structure(pipeline, EXPECTED_FEATURE_COUNT)
                return candidate
            except Exception as exc:
                candidate_errors.append(f'{candidate}: {exc}')
                logger.warning('Skipping unusable preprocessing pipeline %s: %s', candidate, exc)

    raise RuntimeError(
        'No fitted preprocessing pipeline found. Tried: '
        + ', '.join(_PIPELINE_CANDIDATES)
        + ('; failures: ' + ' | '.join(candidate_errors) if candidate_errors else '')
    )


def _validate_pipeline_structure(pipeline, expected_feature_count):
    if not isinstance(pipeline, Pipeline):
        raise RuntimeError(f'Loaded preprocessing artifact is not a sklearn Pipeline: {type(pipeline)!r}')

    required_steps = ('feature_aligner', 'imputer', 'clipper', 'scaler')
    missing_steps = [step for step in required_steps if step not in pipeline.named_steps]
    if missing_steps:
        raise RuntimeError(f'Preprocessing pipeline missing required steps: {missing_steps}')

    feature_aligner = pipeline.named_steps['feature_aligner']
    if type(feature_aligner).__name__ != 'FeatureAlignTransformer':
        raise RuntimeError(
            'feature_aligner step is not FeatureAlignTransformer; '
            f'got {type(feature_aligner).__name__}'
        )

    feature_names = list(getattr(feature_aligner, 'feature_names', []) or [])
    if not feature_names:
        raise RuntimeError('FeatureAlignTransformer has no feature_names metadata')

    if len(feature_names) != expected_feature_count:
        raise RuntimeError(
            'Pipeline feature schema count mismatch: '
            f'expected {expected_feature_count}, got {len(feature_names)}'
        )

    # Verify fitted state for transform-only inference.
    try:
        check_is_fitted(pipeline.named_steps['imputer'])
    except Exception as exc:
        raise RuntimeError(f'Imputer in preprocessing pipeline is not fitted: {exc}') from exc

    try:
        check_is_fitted(pipeline.named_steps['scaler'])
    except Exception as exc:
        raise RuntimeError(f'Scaler in preprocessing pipeline is not fitted: {exc}') from exc

    return feature_names


def _validate_model_feature_schema(model_info, pipeline_feature_names, model_label):
    if not model_info:
        return

    n_features = model_info.get('n_features_in_')
    if n_features is not None and int(n_features) != len(pipeline_feature_names):
        raise RuntimeError(
            f'{model_label} feature count ({n_features}) does not match '
            f'pipeline feature count ({len(pipeline_feature_names)})'
        )

    model_feature_names = model_info.get('feature_names')
    if model_feature_names is None:
        return

    model_feature_names = [str(col) for col in model_feature_names]
    if model_feature_names != pipeline_feature_names:
        mismatches = [
            idx
            for idx, (left, right) in enumerate(zip(model_feature_names, pipeline_feature_names))
            if left != right
        ]
        mismatch_msg = ''
        if mismatches:
            idx0 = mismatches[0]
            mismatch_msg = (
                f' first mismatch at index {idx0}: '
                f'model={model_feature_names[idx0]!r}, pipeline={pipeline_feature_names[idx0]!r}'
            )
        raise RuntimeError(f'{model_label} feature ordering mismatches pipeline schema;{mismatch_msg}')


def preprocess_for_inference(
    raw_df,
    model1_info=None,
    model2_info=None,
    pipeline_path=None,
    expected_feature_count=EXPECTED_FEATURE_COUNT,
):
    """Run strict, deterministic inference preprocessing with a fitted pipeline.

    Requirements enforced:
    - transform-only (no fit during inference)
    - exact feature ordering and count
    - no row drops
    - hard-fail on schema/preprocessing/output integrity issues
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise RuntimeError(f'preprocess_for_inference expects pandas.DataFrame, got {type(raw_df)!r}')

    input_rows, input_cols = raw_df.shape
    logger.info('Preprocessing input shape: rows=%d cols=%d', input_rows, input_cols)

    resolved_pipeline_path = _resolve_pipeline_path(pipeline_path)
    logger.info('Using fitted preprocessing pipeline: %s', resolved_pipeline_path)

    try:
        pipeline = _load_joblib_artifact(resolved_pipeline_path)
    except Exception as exc:
        logger.exception('Failed to load preprocessing pipeline from %s', resolved_pipeline_path)
        raise RuntimeError(f'Failed to load preprocessing pipeline: {exc}') from exc

    pipeline_feature_names = _validate_pipeline_structure(pipeline, expected_feature_count)
    logger.info('Pipeline feature names count: %d', len(pipeline_feature_names))

    try:
        feature_names = joblib.load(os.path.join('src', 'models', 'feature_names.pkl'))
    except Exception:
        feature_names = None

    if feature_names is not None:
        pipeline_feature_names = [str(column) for column in list(feature_names)]
        logger.info('Using %d feature names from src/models/feature_names.pkl', len(pipeline_feature_names))

    _validate_model_feature_schema(model1_info, pipeline_feature_names, 'Model1')
    _validate_model_feature_schema(model2_info, pipeline_feature_names, 'Model2')

    try:
        working_df = raw_df.rename(columns=PCAP_TO_CICIDS)
        renamed_count = len(set(raw_df.columns).intersection(set(PCAP_TO_CICIDS.keys())))
        if renamed_count:
            logger.info('Renamed %d incoming columns using PCAP_TO_CICIDS mapping', renamed_count)
    except Exception:
        working_df = raw_df.copy()

    working_df = clean_flow_features(working_df, pipeline_feature_names)

    try:
        transformed = pipeline.transform(working_df)
    except Exception as exc:
        logger.exception('Preprocessing step failed')
        raise RuntimeError(f'Preprocessing step failed: {exc}') from exc

    X = np.asarray(transformed, dtype=np.float32)

    if X.shape[0] != input_rows:
        raise RuntimeError(
            f'Row count changed during preprocessing: input_rows={input_rows}, transformed_rows={X.shape[0]}'
        )

    if X.shape[1] != expected_feature_count:
        raise RuntimeError(
            f'Unexpected transformed feature count: expected={expected_feature_count}, got={X.shape[1]}'
        )

    logger.info('Preprocessing transformed shape: rows=%d cols=%d', X.shape[0], X.shape[1])
    logger.info('Preprocessing feature count: transformed=%d expected=%d', X.shape[1], expected_feature_count)

    return X
