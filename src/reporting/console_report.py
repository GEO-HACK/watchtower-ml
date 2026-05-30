"""Owns all console output for detection results, metrics, latency summary, and per-flow sample display."""

import os
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from inference.predictor import class_name_for_prediction, prediction_confidence


logger = logging.getLogger(__name__)


def print_detection_report(
    capture_path,
    df,
    X,
    preds1,
    preds2,
    proba1,
    proba2,
    if_preds,
    if_attack_scores,
    fused_results,
    fused_scores,
    final_preds,
    escalated,
    class_names,
    latency_summary,
    y_test=None,
    decision_threshold=None,
    model1_path="",
    model2_path="",
    model3_path="",
):
    """Owns all console output. No computation happens here."""
    print("\n" + "=" * 80)
    print("HYBRID DETECTOR: FLOW ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: {len(df)} flows extracted from {os.path.basename(capture_path)}")
    print(f"Features: {X.shape[1]} aligned features")

    print("\n" + "-" * 80)
    print("INDEPENDENT MODEL PREDICTIONS")
    print("-" * 80)

    unique1, counts1 = np.unique(preds1, return_counts=True)
    unique2, counts2 = np.unique(preds2, return_counts=True)

    print(f"\nModel1 ({os.path.basename(model1_path)}):")
    print(f"  Prediction distribution: {dict(zip(map(int, unique1), map(int, counts1)))}")

    print(f"\nModel2 ({os.path.basename(model2_path)}):")
    print(f"  Prediction distribution: {dict(zip(map(int, unique2), map(int, counts2)))}")

    # Model agreement
    agree = np.mean(preds1 == preds2)
    print(f"\nModel Agreement: {agree:.2%} ({int(np.sum(preds1 == preds2))}/{len(preds1)})")

    print("\n" + "-" * 80)
    print("FUSION LAYER RESULTS")
    print("-" * 80)

    for strategy in fused_results:
        fused = fused_results[strategy]
        scores = fused_scores[strategy]
        unique_f, counts_f = np.unique(fused, return_counts=True)
        print(f"\n[{strategy.upper()}]")
        print(f"  Prediction distribution: {dict(zip(map(int, unique_f), map(int, counts_f)))}")
        print(f"  Confidence score range: {scores.min():.4f} - {scores.max():.4f}")

        if y_test is not None:
            if len(y_test) >= len(fused):
                y_test_aligned = y_test[:len(fused)]
                acc = accuracy_score(y_test_aligned, fused)
                f1_macro = f1_score(y_test_aligned, fused, average='macro', zero_division=0)
                print(f"  Accuracy vs Ground Truth: {acc:.4f}")
                print(f"  Macro F1 Score: {f1_macro:.4f}")
            else:
                logger.warning(f"Ground truth has {len(y_test)} samples but generated {len(fused)} flows")

    print("\n" + "-" * 80)
    print("ISOLATION FOREST ESCALATION LAYER")
    print("-" * 80)
    unique_if, counts_if = np.unique(if_preds, return_counts=True)
    unique_final, counts_final = np.unique(final_preds, return_counts=True)
    print(f"\nModel3 ({os.path.basename(model3_path)}):")
    print(f"  Prediction distribution: {dict(zip(map(int, unique_if), map(int, counts_if)))}")
    print(f"  Normalized anomaly score range: {if_attack_scores.min():.4f} - {if_attack_scores.max():.4f}")
    print(f"  Escalated flows: {int(np.sum(escalated))}")
    print(f"\nFinal output after IF escalation:")
    print(f"  Prediction distribution: {dict(zip(map(int, unique_final), map(int, counts_final)))}")

    if y_test is not None and len(y_test) >= len(fused):
        print("\n" + "-" * 80)
        print("DETAILED CLASSIFICATION METRICS (vs Ground Truth)")
        print("-" * 80)
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

    print("\n" + "-" * 80)
    print("SAMPLE FLOW DETECTIONS (First 10)")
    print("-" * 80)
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

    print("\n" + "=" * 80)

    print("\n" + "-" * 80)
    print("LATENCY SUMMARY (ms)")
    print("-" * 80)
    print(f"  Preprocessing:   {latency_summary['preprocessing']:.4f}")
    print(f"  RF Inference:    {latency_summary['rf_inference']:.4f}")
    print(f"  XGB Inference:   {latency_summary['xgb_inference']:.4f}")
    print(f"  IF Inference:    {latency_summary['if_inference']:.4f}")
    print(f"  Total:           {latency_summary['total']:.4f}")
