import os
import sys
import argparse
import joblib
import logging
import numpy as np
import pandas as pd

from preprocessing.flow_aggregator import FlowAggregator
from preprocessing.preprocessing_pipeline1 import preprocess_for_inference
from utils.latency_tracker import LatencyTracker
from diagnostics.schema_checker import inspect_model_file
from sklearn.metrics import classification_report, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_CANDIDATES = [
    'label',
    'Label',
    'class',
    'Class',
    'attack',
    'Attack',
    'target',
    'Target',
    'y',
]

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
    return model, obj


def predict_with_model(model, X):
    preds = model.predict(X)          # hard predictions — use model directly
    proba = None
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)  # only for fusion scoring
    except Exception as e:
        logger.warning('predict_proba failed: %s', e)
    return preds, proba


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


def prediction_confidence(preds, proba):
    if proba is None or len(preds) == 0:
        return np.ones(len(preds), dtype=float)

    pred_indices = np.argmax(proba, axis=1)
    return proba[np.arange(len(preds)), pred_indices]


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


def class_name_for_prediction(prediction, class_names):
    try:
        prediction_index = int(prediction)
        if 0 <= prediction_index < len(class_names):
            return class_names[prediction_index]
    except Exception:
        pass
    return str(prediction)


def pcap_to_flow_features(pcap_path, max_packets=None, skip_packets=0):
    """Read a PCAP file and return a pandas.DataFrame of flow feature dicts."""
    from preprocessing.packet_capture import capture_from_pcap, get_packet_metadata

    aggregator = FlowAggregator()

    def handle_packet(packet):
        metadata = get_packet_metadata(packet)
        if metadata:
            aggregator.process_packet(metadata)

    capture_from_pcap(
        pcap_path,
        handle_packet,
        max_packets=max_packets,
        skip_packets=skip_packets,
    )
    aggregator.flush()
    flows = aggregator.get_completed_flows()
    if not flows:
        return pd.DataFrame()
    df = pd.DataFrame(flows)
    return df


def csv_to_flow_features(csv_path):
    """Read a CSV capture file and return features plus optional labels.

    The CSV may already contain aggregated flow features. If a label-like column
    is present, it will be extracted and returned separately.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f'CSV capture is empty: {csv_path}')

    label_col = next((col for col in LABEL_CANDIDATES if col in df.columns), None)
    y_test = None
    if label_col is not None:
        y_test = pd.to_numeric(df[label_col], errors='coerce').fillna(0).to_numpy()
        df = df.drop(columns=[label_col])

    return df, y_test


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
    score1 = attack_score_from_proba(proba1)
    score2 = attack_score_from_proba(proba2)

    if score1 is None:
        score1 = np.zeros(len(preds1), dtype=float)
    if score2 is None:
        score2 = np.zeros(len(preds2), dtype=float)

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


def main():
    parser = argparse.ArgumentParser(description='Run hybrid detection on flow features extracted from a PCAP or CSV capture file')
    parser.add_argument('--capture-file', '--pcap', dest='capture_file', default=os.path.join('src', 'data', 'ddos_unlabeled.csv'), help='Path to input capture file (.pcap or .csv)')
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

    # Setup paths
    capture_path = args.capture_file
    model1_path = os.path.join('src', 'models', 'random_forest.pkl')
    model2_path = os.path.join('src', 'models', 'xgboost_model.pkl')
    pipeline_path = os.path.join(SCRIPT_DIR, 'models', 'preprocessing_pipeline.pkl')
    y_test_path = os.path.join('src', 'data', 'y_test.npy')

    if not os.path.exists(capture_path):
        logger.error('Capture file not found: %s', capture_path)
        sys.exit(1)

    if args.capture_format == 'auto':
        capture_format = 'csv' if capture_path.lower().endswith('.csv') else 'pcap'
    else:
        capture_format = args.capture_format

    # Convert capture file to flows/features
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
        # Check if y_test exists (with possible space in filename)
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

        logger.info(
            'Converting PCAP to flows (skip=%d, max=%s)...',
            args.skip_packets,
            str(args.max_packets) if args.max_packets is not None else 'all',
        )
        df = pcap_to_flow_features(
            capture_path,
            max_packets=args.max_packets,
            skip_packets=args.skip_packets,
        )
    logger.info('Flows generated: %d', len(df))

    # By default process the full dataset. Set DEBUG_SAMPLE=True to enable sampling.
    DEBUG_SAMPLE = True
    SAMPLE_SIZE = 100000  # change as needed when DEBUG_SAMPLE=True

    if DEBUG_SAMPLE:
        if len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
            logger.info(f"DEBUG MODE: sampled {SAMPLE_SIZE} random flows from dataset")
        else:
            logger.info("Dataset smaller than sample size; using full dataset")

    if df.empty:
        logger.error('No flows extracted from PCAP')
        sys.exit(1)

    latency_tracker = LatencyTracker()

    # Load models
    logger.info('Loading models...')
    model1, raw1 = load_maybe_dict_model(model1_path)
    model2, raw2 = load_maybe_dict_model(model2_path)
    model3_path = os.path.join('src', 'models', 'isolation_forest.pkl')
    iso_forest = joblib.load(model3_path)
    class_names = get_class_names(model1)

    logger.info('Using class labels: %s', ', '.join(class_names))

    # Inspect models
    model1_info = inspect_model_file(model1_path)
    model2_info = inspect_model_file(model2_path)

    logger.info('Model1 (%s): %s features', os.path.basename(model1_path), model1_info.get('n_features_in_'))
    logger.info('Model2 (%s): %s features', os.path.basename(model2_path), model2_info.get('n_features_in_'))
    logger.info('Model3 (%s): %s features', os.path.basename(model3_path), getattr(iso_forest, 'n_features_in_', None))

    try:
        latency_tracker.start('preprocessing')
        X = preprocess_for_inference(
            df,
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

    # Run inference with both models
    logger.info('Running Model1 inference...')
    latency_tracker.start('rf_inference')
    preds1, proba1 = predict_with_model(model1, X)
    latency_tracker.stop('rf_inference')
    
    logger.info('Running Model2 inference...')
    latency_tracker.start('xgb_inference')
    preds2, proba2 = predict_with_model(model2, X)
    latency_tracker.stop('xgb_inference')

    logger.info('Running Model3 (Isolation Forest) inference...')
    latency_tracker.start('if_inference')
    if_preds = iso_forest.predict(X)         # +1 normal, -1 anomaly
    if_scores = iso_forest.score_samples(X)   # more negative = more anomalous
    latency_tracker.stop('if_inference')

    # Normalize IF score to [0, 1] attack range for consistent display
    # score_samples returns negative floats — flip so 1.0 = most anomalous
    if_score_min = if_scores.min()
    if_score_max = if_scores.max()
    if_attack_scores = 1.0 - (
        (if_scores - if_score_min) / (if_score_max - if_score_min + 1e-9)
    )

    # Apply fusion strategies
    strategies = ["majority", "or", "confidence_weighted", "unanimous_or_majority"]
    fused_results = {}
    
    logger.info('Applying fusion strategies...')
    for strategy in strategies:
        fused_results[strategy] = combine_predictions(preds1, preds2, proba1, proba2, strategy=strategy)

    fused_scores = {
        strategy: combine_scores(preds1, preds2, proba1, proba2, strategy=strategy)
        for strategy in strategies
    }

    # IF escalation — only activates on flows the supervised models cleared
    final_preds = fused_results['majority'].copy().astype(int)
    escalated = np.zeros(len(final_preds), dtype=bool)

    for i in range(len(final_preds)):
        rf_xgb_said_benign = final_preds[i] == 0
        if_said_anomaly = if_preds[i] == -1

        if rf_xgb_said_benign and if_said_anomaly:
            final_preds[i] = -1   # -1 = ANOMALY (unknown threat type)
            escalated[i] = True

    n_escalated = escalated.sum()
    logger.info(
        'IF escalation: %d flows escalated from BENIGN to ANOMALY',
        n_escalated
    )

    decision_threshold = args.decision_threshold
    optimized_threshold = None
    optimized_f1 = None
    if args.optimize_threshold:
        if y_test is None:
            logger.warning('Threshold optimization requested but no ground truth labels are available')
        else:
            optimized_threshold, optimized_f1 = optimize_threshold(
                y_test[:len(fused_scores['majority'])],
                fused_scores['majority'],
            )
            decision_threshold = optimized_threshold
            logger.info(
                'Optimized attack-confidence threshold: %.4f (binary F1=%.4f)',
                optimized_threshold,
                optimized_f1,
            )

    # Optionally export misclassified flows to PCAPs
    if args.export_misclassified:
        if capture_format == 'csv':
            logger.warning('Misclassified flow export is only available for PCAP input; skipping export')
        elif y_test is None:
            logger.error('Cannot export misclassified flows: ground truth (y_test) not available')
        else:
            from preprocessing.packet_capture import save_packets_for_flows

            # choose prediction vector
            if args.export_which == 'model1':
                chosen_preds = preds1
            elif args.export_which == 'model2':
                chosen_preds = preds2
            else:
                # fused majority
                chosen_preds = fused_results['majority']

            # align y_test length with preds
            y_aligned = y_test[:len(chosen_preds)]
            mis_idx = np.where(chosen_preds != y_aligned)[0]
            logger.info('Found %d misclassified flows (exporting to %s)', len(mis_idx), args.export_misclassified)
            if len(mis_idx) > 0:
                flows_records = df.to_dict('records')
                results = save_packets_for_flows(capture_path, flows_records, mis_idx, args.export_misclassified, pad_seconds=args.pad_seconds)
                logger.info('Exported packets for %d flows', len([v for v in results.values() if v > 0]))

    # Display results
    print("\n" + "="*80)
    print("HYBRID DETECTOR: FLOW ANALYSIS")
    print("="*80)
    print(f"\nDataset: {len(df)} flows extracted from {os.path.basename(capture_path)}")
    print(f"Features: {X.shape[1]} aligned features")

    print("\n" + "-"*80)
    print("INDEPENDENT MODEL PREDICTIONS")
    print("-"*80)
    
    unique1, counts1 = np.unique(preds1, return_counts=True)
    unique2, counts2 = np.unique(preds2, return_counts=True)
    
    print(f"\nModel1 ({os.path.basename(model1_path)}):")
    print(f"  Prediction distribution: {dict(zip(map(int, unique1), map(int, counts1)))}")
    
    print(f"\nModel2 ({os.path.basename(model2_path)}):")
    print(f"  Prediction distribution: {dict(zip(map(int, unique2), map(int, counts2)))}")
    
    # Model agreement
    agree = np.mean(preds1 == preds2)
    print(f"\nModel Agreement: {agree:.2%} ({int(np.sum(preds1 == preds2))}/{len(preds1)})")

    print("\n" + "-"*80)
    print("FUSION LAYER RESULTS")
    print("-"*80)
    
    for strategy in strategies:
        fused = fused_results[strategy]
        scores = fused_scores[strategy]
        unique_f, counts_f = np.unique(fused, return_counts=True)
        print(f"\n[{strategy.upper()}]")
        print(f"  Prediction distribution: {dict(zip(map(int, unique_f), map(int, counts_f)))}")
        print(f"  Confidence score range: {scores.min():.4f} - {scores.max():.4f}")
        
        if y_test is not None:
            # Check sample size match
            if len(y_test) >= len(fused):
                y_test_aligned = y_test[:len(fused)]
                acc = accuracy_score(y_test_aligned, fused)
                f1_macro = f1_score(y_test_aligned, fused, average='macro', zero_division=0)
                print(f"  Accuracy vs Ground Truth: {acc:.4f}")
                print(f"  Macro F1 Score: {f1_macro:.4f}")
            else:
                logger.warning(f"Ground truth has {len(y_test)} samples but generated {len(fused)} flows")

    print("\n" + "-"*80)
    print("ISOLATION FOREST ESCALATION LAYER")
    print("-"*80)
    unique_if, counts_if = np.unique(if_preds, return_counts=True)
    unique_final, counts_final = np.unique(final_preds, return_counts=True)
    print(f"\nModel3 ({os.path.basename(model3_path)}):")
    print(f"  Prediction distribution: {dict(zip(map(int, unique_if), map(int, counts_if)))}")
    print(f"  Normalized anomaly score range: {if_attack_scores.min():.4f} - {if_attack_scores.max():.4f}")
    print(f"  Escalated flows: {int(n_escalated)}")
    print(f"\nFinal output after IF escalation:")
    print(f"  Prediction distribution: {dict(zip(map(int, unique_final), map(int, counts_final)))}")

    # Detailed metrics if ground truth available
    if y_test is not None and len(y_test) >= len(fused):
        print("\n" + "-"*80)
        print("DETAILED CLASSIFICATION METRICS (vs Ground Truth)")
        print("-"*80)
        
        y_test_aligned = y_test[:len(fused)]
        
        print(f"\nModel1 Classification Report:")
        print(classification_report(y_test_aligned, preds1, labels=list(range(len(class_names))), target_names=class_names, zero_division=0))
        
        print(f"\nModel2 Classification Report:")
        print(classification_report(y_test_aligned, preds2, labels=list(range(len(class_names))), target_names=class_names, zero_division=0))
        
        print(f"\nFused (MAJORITY) Classification Report:")
        print(classification_report(y_test_aligned, fused_results["majority"], labels=list(range(len(class_names))), target_names=class_names, zero_division=0))

        if decision_threshold is not None:
            fused_attack_binary = (fused_scores['majority'] >= decision_threshold).astype(int)
            true_attack_binary = (y_test_aligned != 0).astype(int)
            attack_acc = accuracy_score(true_attack_binary, fused_attack_binary)
            attack_f1 = f1_score(true_attack_binary, fused_attack_binary, zero_division=0)
            print(f"\nAttack/Benign Threshold: {decision_threshold:.4f}")
            print(f"Attack-vs-Benign Accuracy: {attack_acc:.4f}")
            print(f"Attack-vs-Benign F1: {attack_f1:.4f}")

    # Sample predictions
    print("\n" + "-"*80)
    print("SAMPLE FLOW DETECTIONS (First 10)")
    print("-"*80)
    
    n_samples = min(10, len(df))
    for i in range(n_samples):
        row = df.iloc[i].to_dict()
        m1_name = class_name_for_prediction(preds1[i], class_names)
        m2_name = class_name_for_prediction(preds2[i], class_names)
        fused_name = class_name_for_prediction(fused_results['majority'][i], class_names)
        final_name = class_name_for_prediction(final_preds[i], class_names)
        m1_score = float(prediction_confidence(np.asarray([preds1[i]]), proba1[i:i+1] if proba1 is not None else None)[0])
        m2_score = float(prediction_confidence(np.asarray([preds2[i]]), proba2[i:i+1] if proba2 is not None else None)[0])
        fused_score = float(fused_scores['majority'][i])
        if_score = float(if_attack_scores[i])
        decision = fused_name if fused_name != 'BENIGN' else 'BENIGN'
        final_decision = final_name if final_name != 'BENIGN' else 'BENIGN'
        if decision_threshold is not None and fused_score < decision_threshold:
            decision = f'BENIGN (low attack score)' if fused_name == 'BENIGN' else f'{fused_name} (below threshold)'
        print(f"\n[Flow {i+1}]")
        print(f"  {row.get('src_ip')} : {row.get('src_port')} -> {row.get('dst_ip')} : {row.get('dst_port')}")
        print(f"  Model1: {int(preds1[i])} ({m1_name}, score={m1_score:.4f})")
        print(f"  Model2: {int(preds2[i])} ({m2_name}, score={m2_score:.4f})")
        print(f"  Fused:  {int(fused_results['majority'][i])} ({fused_name}, attack_score={fused_score:.4f}, threshold={decision_threshold if decision_threshold is not None else 'n/a'}, decision={decision})")
        print(f"  IF:     {int(if_preds[i])} (anomaly_score={if_score:.4f})")
        print(f"  Final:  {int(final_preds[i])} ({final_name}, escalated={bool(escalated[i])}, decision={final_decision})")
        if y_test is not None and i < len(y_test):
            print(f"  Ground Truth: {int(y_test[i])} ({class_name_for_prediction(y_test[i], class_names)})")

    print("\n" + "="*80)

    latency_summary = latency_tracker.summary()

    print("\n" + "-"*80)
    print("LATENCY SUMMARY (ms)")
    print("-"*80)
    print(f"  Preprocessing:   {latency_summary['preprocessing']:.4f}")
    print(f"  RF Inference:    {latency_summary['rf_inference']:.4f}")
    print(f"  XGB Inference:   {latency_summary['xgb_inference']:.4f}")
    print(f"  IF Inference:    {latency_summary['if_inference']:.4f}")
    print(f"  Total:           {latency_summary['total']:.4f}")

    return {
        'preprocessing_latency_ms': latency_summary['preprocessing'],
        'rf_inference_latency_ms': latency_summary['rf_inference'],
        'xgb_inference_latency_ms': latency_summary['xgb_inference'],
        'if_inference_latency_ms': latency_summary['if_inference'],
        'total_latency_ms': latency_summary['total'],
    }


if __name__ == '__main__':
    main()
