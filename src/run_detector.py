"""Thin, logic-free orchestrator for the hybrid detector.

This module imports only from the refactored modules and delegates all
computation to them. It preserves the original CLI interface and return
payload shape (including latency keys).
"""
import os
import sys
import argparse
import joblib
import logging
import numpy as np

from inference.model_loader import load_maybe_dict_model, get_class_names, warmup_models
from inference.predictor import prepare_input
from inference.fusion import combine_predictions, combine_scores
from inference.escalation import optimize_threshold
from ingestion.capture_reader import pcap_to_flow_features, csv_to_flow_features, LABEL_CANDIDATES
from reporting.console_report import print_detection_report
from utils.latency_tracker import LatencyTracker
from diagnostics.schema_checker import inspect_model_file
from preprocessing.preprocessing_pipeline1 import preprocess_for_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run hybrid detection on flow features extracted from a PCAP or CSV capture file')
    parser.add_argument('--capture-file', '--pcap', dest='capture_file', default=os.path.join('src', 'data', 'watchtower_blind_test_flows.csv'), help='Path to input capture file (.pcap or .csv)')
    parser.add_argument('--capture-format', choices=['auto', 'pcap', 'csv'], default='auto', help='Override capture file type detection')
    parser.add_argument('--max-packets', type=int, default=1000, help='Process only this many packets from the PCAP for faster testing (default: 1000)')
    parser.add_argument('--skip-packets', type=int, default=0, help='Skip this many initial packets before processing')
    parser.add_argument('--y-test', default=None, help='Optional path to labels for CSV capture files')
    parser.add_argument('--export-misclassified', default=None, help='Directory to write per-flow PCAPs for misclassified flows')
    parser.add_argument('--export-which', choices=['model1','model2','fused'], default='model1', help='Which predictions to consider when exporting misclassified flows')
    parser.add_argument('--pad-seconds', type=float, default=0.5, help='Seconds to pad flow time window when exporting packets')
    parser.add_argument('--decision-threshold', type=float, default=None, help='Minimum merged confidence required before accepting an attack prediction')
    parser.add_argument('--optimize-threshold', action='store_true', help='Optimize the merged confidence threshold against available ground truth labels')
    args = parser.parse_args()

    if args.max_packets is not None and args.max_packets <= 0:
        logger.error('--max-packets must be > 0 when provided')
        sys.exit(1)

    if args.skip_packets < 0:
        logger.error('--skip-packets must be >= 0')
        sys.exit(1)

    capture_path = args.capture_file
    model1_path = os.path.join('src', 'models', 'random_forest.pkl')
    model2_path = os.path.join('src', 'models', 'xgboost_model.pkl')
    model3_path = os.path.join('src', 'models', 'isolation_forest.pkl')
    pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'preprocessing_pipeline.pkl')
    y_test_path = os.path.join('src', 'data', 'y_test.npy')

    if not os.path.exists(capture_path):
        logger.error('Capture file not found: %s', capture_path)
        sys.exit(1)

    capture_format = 'csv' if capture_path.lower().endswith('.csv') else 'pcap' if args.capture_format == 'auto' else args.capture_format

    # Read input capture
    y_test = None
    if capture_format == 'csv':
        logger.info('Loading flow features from CSV: %s', capture_path)
        df, embedded_y = csv_to_flow_features(capture_path)
        if args.y_test is not None:
            y_test_path = args.y_test
            if os.path.exists(y_test_path):
                y_test = np.load(y_test_path)
                logger.info('Loaded ground truth labels from %s: %d samples', y_test_path, len(y_test))
            else:
                logger.warning('Provided y_test file not found: %s', y_test_path)
        elif embedded_y is not None:
            y_test = embedded_y
            logger.info('Loaded labels from CSV capture file: %d samples', len(y_test))
        else:
            logger.warning('CSV capture has no labels; will show predictions only')
    else:
        if args.y_test is not None:
            y_test_path = args.y_test
        elif not os.path.exists(y_test_path):
            alt_y_test_path = os.path.join('src', 'data', 'y_test .npy')
            if os.path.exists(alt_y_test_path):
                y_test_path = alt_y_test_path

        if os.path.exists(y_test_path):
            y_test = np.load(y_test_path)
            logger.info('Loaded ground truth labels: %d samples', len(y_test))
        else:
            logger.warning('No ground truth labels found; will show predictions only')

        logger.info('Converting PCAP to flows (skip=%d, max=%s)...', args.skip_packets, str(args.max_packets) if args.max_packets is not None else 'all')
        df = pcap_to_flow_features(capture_path, max_packets=args.max_packets, skip_packets=args.skip_packets)

    logger.info('Flows generated: %d', len(df))

    DEBUG_SAMPLE = True
    SAMPLE_SIZE = 100000
    if DEBUG_SAMPLE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        logger.info('DEBUG MODE: sampled %d random flows from dataset', SAMPLE_SIZE)
    else:
        logger.info('Dataset smaller than sample size; using full dataset')

    if df.empty:
        logger.error('No flows extracted from capture')
        sys.exit(1)

    latency_tracker = LatencyTracker()

    logger.info('Loading models...')
    model1, raw1 = load_maybe_dict_model(model1_path)
    # runtime params handled by loader

    model2, raw2 = load_maybe_dict_model(model2_path)

    iso_forest = joblib.load(model3_path)

    class_names = get_class_names(model1)
    logger.info('Using class labels: %s', ', '.join(class_names))

    model1_info = inspect_model_file(model1_path)
    model2_info = inspect_model_file(model2_path)

    logger.info('Model1 (%s): %s features', os.path.basename(model1_path), model1_info.get('n_features_in_'))
    logger.info('Model2 (%s): %s features', os.path.basename(model2_path), model2_info.get('n_features_in_'))
    logger.info('Model3 (%s): %s features', os.path.basename(model3_path), getattr(iso_forest, 'n_features_in_', None))

    try:
        n_features = model1_info.get('n_features_in_') or model2_info.get('n_features_in_') or getattr(iso_forest, 'n_features_in_', None) or 70
        warmup_models(model1, model2, iso_forest, n_features)
        logger.info('Model warm-up complete')
    except Exception as e:
        logger.warning('Model warm-up failed: %s', e)

    try:
        model_input = prepare_input(df)
        latency_tracker.start('preprocessing')
        X = preprocess_for_inference(
            model_input,
            model1_info=model1_info,
            model2_info=model2_info,
            pipeline_path=pipeline_path,
            expected_feature_count=70,
        )
        latency_tracker.stop('preprocessing')
    except RuntimeError as e:
        logger.error('Inference preprocessing failed: %s', e)
        sys.exit(1)

    if X is None or getattr(X, 'size', 0) == 0:
        logger.error('No feature data available for inference')
        sys.exit(1)

    logger.info('Shared transformed input shape for both models: %s, dtype=%s', X.shape, X.dtype)

    logger.info('Running Model1 inference...')
    latency_tracker.start('rf_inference')
    proba1 = model1.predict_proba(X)
    preds1 = np.asarray(model1.classes_)[np.argmax(proba1, axis=1)]
    latency_tracker.stop('rf_inference')

    logger.info('Running Model2 inference...')
    latency_tracker.start('xgb_inference')
    proba2 = model2.predict_proba(X)
    preds2 = np.asarray(model2.classes_)[np.argmax(proba2, axis=1)]
    latency_tracker.stop('xgb_inference')

    logger.info('Running Model3 (Isolation Forest) inference...')
    latency_tracker.start('if_inference')
    if_preds = iso_forest.predict(X)
    if_scores = iso_forest.score_samples(X)
    latency_tracker.stop('if_inference')

    if_score_min = if_scores.min()
    if_score_max = if_scores.max()
    if_attack_scores = 1.0 - ((if_scores - if_score_min) / (if_score_max - if_score_min + 1e-9))

    strategies = ["majority", "or", "confidence_weighted", "unanimous_or_majority"]
    fused_results = {s: combine_predictions(preds1, preds2, proba1, proba2, strategy=s) for s in strategies}
    fused_scores = {s: combine_scores(preds1, preds2, proba1, proba2, strategy=s) for s in strategies}

    final_preds = fused_results['confidence_weighted'].copy().astype(int)
    escalated = np.zeros(len(final_preds), dtype=bool)
    for i in range(len(final_preds)):
        rf_xgb_said_benign = final_preds[i] == 0
        if_said_anomaly = if_preds[i] == -1
        if rf_xgb_said_benign and if_said_anomaly:
            final_preds[i] = -1
            escalated[i] = True

    n_escalated = int(escalated.sum())
    logger.info('IF escalation: %d flows escalated from BENIGN to ANOMALY', n_escalated)

    decision_threshold = args.decision_threshold
    optimized_threshold = None
    optimized_f1 = None
    if args.optimize_threshold:
        if y_test is None:
            logger.warning('Threshold optimization requested but no ground truth labels are available')
        else:
            optimized_threshold, optimized_f1 = optimize_threshold(y_test[:len(fused_scores['majority'])], fused_scores['majority'])
            decision_threshold = optimized_threshold
            logger.info('Optimized attack-confidence threshold: %.4f (binary F1=%.4f)', optimized_threshold, optimized_f1)

    # Delegate all printing to the reporting module
    latency_summary = latency_tracker.summary()
    print_detection_report(
        capture_path=capture_path,
        df=df,
        X=X,
        preds1=preds1,
        preds2=preds2,
        proba1=proba1,
        proba2=proba2,
        if_preds=if_preds,
        if_attack_scores=if_attack_scores,
        fused_results=fused_results,
        fused_scores=fused_scores,
        final_preds=final_preds,
        escalated=escalated,
        class_names=class_names,
        latency_summary=latency_summary,
        y_test=y_test,
        decision_threshold=decision_threshold,
        model1_path=model1_path,
        model2_path=model2_path,
        model3_path=model3_path,
    )

    return {
        'preprocessing_latency_ms': latency_summary['preprocessing'],
        'rf_inference_latency_ms': latency_summary['rf_inference'],
        'xgb_inference_latency_ms': latency_summary['xgb_inference'],
        'if_inference_latency_ms': latency_summary['if_inference'],
        'total_latency_ms': latency_summary['total'],
    }


if __name__ == '__main__':
    main()
