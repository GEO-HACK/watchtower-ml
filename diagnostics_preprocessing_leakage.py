#!/usr/bin/env python3
"""
Diagnostic script to identify preprocessing data leakage issues.

Usage:
    python diagnostics_preprocessing_leakage.py
    python diagnostics_preprocessing_leakage.py --labeled-data src/data/labeled_test.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-10s | %(message)s'
)
logger = logging.getLogger(__name__)


def check_pipeline_existence():
    """Check if preprocessing pipeline file exists."""
    logger.info("=" * 70)
    logger.info("CHECK 1: Preprocessing Pipeline File Existence")
    logger.info("=" * 70)
    
    candidates = [
        'src/models/preprocessing_pipeline.pkl',
        'src/preprocessing/preprocessing_pipeline.pkl',
        'src/preprocessing/preprocessing_pipeline1.pkl',
    ]
    
    found = False
    for path in candidates:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"✅ Found: {path} ({size_mb:.2f} MB)")
            found = True
        else:
            logger.info(f"❌ Missing: {path}")
    
    if not found:
        logger.error("🔴 CRITICAL: No preprocessing pipeline found!")
        logger.error("   → Models cannot scale inference data correctly")
        return None
    
    return next(p for p in candidates if os.path.exists(p))


def analyze_pipeline_structure(pipeline_path):
    """Analyze the structure of the loaded pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("CHECK 2: Pipeline Structure and Scaler Statistics")
    logger.info("=" * 70)
    
    try:
        pipeline = joblib.load(pipeline_path)
    except Exception as e:
        logger.error(f"❌ Failed to load pipeline: {e}")
        return None
    
    if not hasattr(pipeline, 'named_steps'):
        logger.error("❌ Pipeline is not a sklearn Pipeline")
        return None
    
    logger.info(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
    
    # Check scaler
    if 'scaler' not in pipeline.named_steps:
        logger.error("❌ Pipeline missing 'scaler' step")
        return None
    
    scaler = pipeline.named_steps['scaler']
    logger.info(f"Scaler type: {type(scaler).__name__}")
    
    # Different scalers have different attributes
    if hasattr(scaler, 'center_'):
        logger.info(f"Scaler center (first 5): {scaler.center_[:5] if len(scaler.center_) > 0 else 'N/A'}")
        logger.info(f"Scaler scale (first 5): {scaler.scale_[:5] if len(scaler.scale_) > 0 else 'N/A'}")
    elif hasattr(scaler, 'mean_'):
        logger.info(f"Scaler mean (first 5): {scaler.mean_[:5] if len(scaler.mean_) > 0 else 'N/A'}")
        logger.info(f"Scaler std (first 5): {scaler.scale_[:5] if len(scaler.scale_) > 0 else 'N/A'}")
    
    # Check feature count
    if 'feature_aligner' in pipeline.named_steps:
        aligner = pipeline.named_steps['feature_aligner']
        feature_names = getattr(aligner, 'feature_names', [])
        logger.info(f"Feature count from pipeline: {len(feature_names)}")
        if len(feature_names) < 10:
            logger.info(f"Feature names: {feature_names}")
        else:
            logger.info(f"First 5 features: {feature_names[:5]}")
            logger.info(f"Last 5 features: {feature_names[-5:]}")
    
    return pipeline


def analyze_labeled_data(data_path, pipeline):
    """Analyze preprocessing impact on labeled data."""
    logger.info("\n" + "=" * 70)
    logger.info("CHECK 3: Preprocessing Impact on Labeled Data")
    logger.info("=" * 70)
    
    if not os.path.exists(data_path):
        logger.warning(f"⚠️  Data file not found: {data_path}")
        return
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return
    
    # Remove label column if present
    label_candidates = ['label', 'Label', 'class', 'Class', 'attack', 'Attack']
    label_col = next((col for col in label_candidates if col in df.columns), None)
    
    if label_col:
        logger.info(f"Found label column: '{label_col}'")
        y = pd.to_numeric(df[label_col], errors='coerce').fillna(0).to_numpy()
        df = df.drop(columns=[label_col])
        logger.info(f"Label distribution: {np.bincount(y.astype(int))}")
    else:
        logger.warning("⚠️  No label column found - assuming only features")
        y = None
    
    # BEFORE preprocessing
    logger.info("\n--- BEFORE Preprocessing ---")
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        logger.info(f"First 5 features in raw data:")
        for col in list(numeric_cols)[:5]:
            values = df_numeric[col].dropna()
            if len(values) > 0:
                logger.info(f"  {col:30} | mean={values.mean():12.4f} std={values.std():12.4f} min={values.min():12.4f} max={values.max():12.4f}")
    
    # AFTER preprocessing
    if pipeline:
        logger.info("\n--- AFTER Preprocessing ---")
        try:
            X_processed = pipeline.transform(df_numeric)
            logger.info(f"Processed shape: {X_processed.shape}")
            logger.info(f"First 5 features after preprocessing:")
            for i in range(min(5, X_processed.shape[1])):
                col_data = X_processed[:, i]
                logger.info(f"  Feature[{i:2d}]            | mean={col_data.mean():12.4f} std={col_data.std():12.4f} min={col_data.min():12.4f} max={col_data.max():12.4f}")
            
            # Check for data quality issues
            nan_count = np.isnan(X_processed).sum()
            inf_count = np.isinf(X_processed).sum()
            if nan_count > 0:
                logger.error(f"🔴 Found {nan_count} NaN values in processed data!")
            if inf_count > 0:
                logger.error(f"🔴 Found {inf_count} Inf values in processed data!")
            if nan_count == 0 and inf_count == 0:
                logger.info(f"✅ No NaN or Inf values")
            
            # Check if features are suspiciously uniform
            if np.any(X_processed.std(axis=0) < 0.01):
                logger.error("🔴 Some features have very low std (< 0.01) - possible data collapse!")
            
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")


def analyze_model_predictions(pipeline, data_path, model_path):
    """Test model predictions on labeled data."""
    logger.info("\n" + "=" * 70)
    logger.info("CHECK 4: Model Prediction Behavior")
    logger.info("=" * 70)
    
    if not os.path.exists(model_path):
        logger.warning(f"⚠️  Model not found: {model_path}")
        return
    
    if not os.path.exists(data_path):
        logger.warning(f"⚠️  Data not found: {data_path}")
        return
    
    try:
        df = pd.read_csv(data_path)
        label_candidates = ['label', 'Label', 'class', 'Class', 'attack', 'Attack']
        label_col = next((col for col in label_candidates if col in df.columns), None)
        
        if label_col:
            y_true = pd.to_numeric(df[label_col], errors='coerce').fillna(0).to_numpy().astype(int)
            df = df.drop(columns=[label_col])
        else:
            logger.warning("⚠️  No labels found - cannot evaluate")
            return
        
        # Preprocess
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        X = pipeline.transform(df_numeric)
        
        # Load and predict
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        
        logger.info(f"Predictions distribution: {np.bincount(y_pred)}")
        logger.info(f"True labels distribution: {np.bincount(y_true)}")
        
        # Check for collapse to single class
        unique_preds = len(np.unique(y_pred))
        logger.info(f"Unique predictions: {unique_preds}")
        
        if unique_preds == 1:
            logger.error(f"🔴 CRITICAL: Model predicts only class {y_pred[0]} for all samples!")
            logger.error("   → This indicates preprocessing mismatch or scaler problems")
        
        # Accuracy
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        
        if acc < 0.3:
            logger.error("🔴 Very low accuracy - likely preprocessing problem")
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose preprocessing data leakage')
    parser.add_argument('--labeled-data', type=str, help='Path to labeled test CSV')
    parser.add_argument('--model', type=str, default='src/models/random_forest.pkl', help='Path to model')
    
    args = parser.parse_args()
    
    logger.info("\n")
    logger.info("🔍 PREPROCESSING LEAKAGE DIAGNOSTIC")
    logger.info("Analyzing Watchtower ML preprocessing pipeline\n")
    
    # Check 1: File existence
    pipeline_path = check_pipeline_existence()
    if not pipeline_path:
        logger.error("\n❌ FATAL: Preprocessing pipeline not found!")
        sys.exit(1)
    
    # Check 2: Pipeline structure
    pipeline = analyze_pipeline_structure(pipeline_path)
    
    # Check 3: Optional data analysis
    if args.labeled_data:
        analyze_labeled_data(args.labeled_data, pipeline)
    
    # Check 4: Optional model evaluation
    if args.labeled_data and os.path.exists(args.model):
        analyze_model_predictions(pipeline, args.labeled_data, args.model)
    
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("See PREPROCESSING_DATA_LEAKAGE_ANALYSIS.md for detailed explanation")
    logger.info("and corrective action steps.\n")


if __name__ == '__main__':
    main()
