#!/usr/bin/env python3
"""
Rebuild the preprocessing pipeline with CORRECT data leakage prevention.

This script:
1. Loads training data (must be properly split from test data)
2. Creates preprocessing pipeline fitted ONLY on training data
3. Saves the pipeline for use in inference
4. Validates the pipeline on both training and test data

CRITICAL: The input X_train must be data that was NEVER used for validation/test!
"""

import os
import sys
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Import our custom transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing.custom_transformers import build_preprocessing_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_correct_pipeline(X_train, output_path='src/models/preprocessing_pipeline.pkl'):
    """
    Build preprocessing pipeline fitted ONLY on training data.
    
    CRITICAL: X_train must NOT include any test/validation/future-inference data!
    
    Args:
        X_train: Training features only (DataFrame or array)
        output_path: Where to save the pipeline
    
    Returns:
        (pipeline, feature_names)
    """
    logger.info("=" * 70)
    logger.info("STEP 1: Building Preprocessing Pipeline")
    logger.info("=" * 70)
    
    if isinstance(X_train, np.ndarray):
        logger.warning("⚠️  X_train is a numpy array - cannot determine feature names")
        logger.warning("   Loading from feature_order.pkl if available...")
        
        try:
            feature_names = joblib.load('src/models/feature_order.pkl')
            X_train = pd.DataFrame(X_train, columns=feature_names)
            logger.info(f"✅ Loaded feature names: {len(feature_names)} features")
        except Exception as e:
            logger.error(f"❌ Cannot determine feature names: {e}")
            sys.exit(1)
    
    if not isinstance(X_train, pd.DataFrame):
        logger.error("❌ X_train must be pandas DataFrame or loadable as one")
        sys.exit(1)
    
    feature_names = list(X_train.columns)
    logger.info(f"Features: {len(feature_names)} total")
    logger.info(f"  First 5: {feature_names[:5]}")
    logger.info(f"  Last 5: {feature_names[-5:]}")
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Training data memory: {X_train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Build pipeline
    logger.info("\nBuilding pipeline steps...")
    pipeline = build_preprocessing_pipeline(feature_names)
    logger.info(f"✅ Pipeline created with steps: {list(pipeline.named_steps.keys())}")
    
    # Fit on training data ONLY
    logger.info(f"\nFitting pipeline on {len(X_train)} training samples...")
    logger.info("⚠️  CRITICAL: This must be training data only!")
    
    # Convert to numeric with error handling
    X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce')
    
    # Report missing values before fitting
    nan_count = X_train_numeric.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"⚠️  Found {nan_count} NaN values in training data")
        logger.info("   These will be imputed by SimpleImputer step")
    
    try:
        pipeline.fit(X_train_numeric)
        logger.info("✅ Pipeline fitted successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fit pipeline: {e}")
        sys.exit(1)
    
    # Save pipeline
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info(f"✅ Pipeline saved to: {output_path}")
    
    return pipeline, feature_names


def validate_pipeline(pipeline, X_train, X_test, y_test=None):
    """Validate pipeline by comparing train vs test statistics."""
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Validate Pipeline on Training vs Test Data")
    logger.info("=" * 70)
    
    # Process train data
    logger.info("\nTransforming training data...")
    X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce')
    try:
        X_train_transformed = pipeline.transform(X_train_numeric)
        logger.info(f"✅ Training data transformed: {X_train_transformed.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to transform training data: {e}")
        return False
    
    # Process test data
    logger.info("\nTransforming test data...")
    if isinstance(X_test, np.ndarray):
        # Convert to DataFrame using same feature names
        X_test_df = pd.DataFrame(X_test, columns=X_train.columns)
    else:
        X_test_df = X_test
    
    X_test_numeric = X_test_df.apply(pd.to_numeric, errors='coerce')
    try:
        X_test_transformed = pipeline.transform(X_test_numeric)
        logger.info(f"✅ Test data transformed: {X_test_transformed.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to transform test data: {e}")
        return False
    
    # Compare statistics
    logger.info("\n--- Statistics Comparison ---")
    logger.info("Train Mean (first 5 features):")
    train_mean = X_train_transformed.mean(axis=0)
    for i in range(min(5, len(train_mean))):
        logger.info(f"  Feature {i}: {train_mean[i]:12.6f}")
    
    logger.info("\nTest Mean (first 5 features):")
    test_mean = X_test_transformed.mean(axis=0)
    for i in range(min(5, len(test_mean))):
        logger.info(f"  Feature {i}: {test_mean[i]:12.6f}")
    
    logger.info("\nTrain Std (first 5 features):")
    train_std = X_train_transformed.std(axis=0)
    for i in range(min(5, len(train_std))):
        logger.info(f"  Feature {i}: {train_std[i]:12.6f}")
    
    logger.info("\nTest Std (first 5 features):")
    test_std = X_test_transformed.std(axis=0)
    for i in range(min(5, len(test_std))):
        logger.info(f"  Feature {i}: {test_std[i]:12.6f}")
    
    # Check for issues
    logger.info("\n--- Data Quality Checks ---")
    
    train_nan = np.isnan(X_train_transformed).sum()
    test_nan = np.isnan(X_test_transformed).sum()
    if train_nan == 0 and test_nan == 0:
        logger.info(f"✅ No NaN values in transformed data")
    else:
        logger.error(f"🔴 NaN found - train: {train_nan}, test: {test_nan}")
    
    train_inf = np.isinf(X_train_transformed).sum()
    test_inf = np.isinf(X_test_transformed).sum()
    if train_inf == 0 and test_inf == 0:
        logger.info(f"✅ No Inf values in transformed data")
    else:
        logger.error(f"🔴 Inf found - train: {train_inf}, test: {test_inf}")
    
    # Check that train mean is close to 0 (for StandardScaler/RobustScaler)
    if np.all(np.abs(train_mean) < 0.5):  # Allow some tolerance
        logger.info(f"✅ Training mean close to 0 (robust scaler behavior)")
    else:
        logger.warning(f"⚠️  Training mean not close to 0 - check scaler type")
    
    # Check that train std is close to 1 or reasonable
    if np.all((train_std > 0.01) & (train_std < 100)):
        logger.info(f"✅ Training std in reasonable range (0.01 - 100)")
    else:
        logger.warning(f"⚠️  Some training features have extreme std values")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild preprocessing pipeline with correct train/test split'
    )
    parser.add_argument(
        '--train-data',
        required=True,
        help='CSV file with training features (ONLY training data, no test/val)'
    )
    parser.add_argument(
        '--test-data',
        help='CSV file with test features (for validation only, not used for fitting)'
    )
    parser.add_argument(
        '--output',
        default='src/models/preprocessing_pipeline.pkl',
        help='Output path for pipeline'
    )
    
    args = parser.parse_args()
    
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("REBUILD PREPROCESSING PIPELINE - DATA LEAKAGE PREVENTION")
    logger.info("=" * 70)
    logger.info("\nCRITICAL REQUIREMENT:")
    logger.info("  Input CSV must contain ONLY training data (never seen at test time)")
    logger.info("  The preprocessing will be fit on this data only")
    logger.info("\n")
    
    # Load training data
    logger.info(f"Loading training data from: {args.train_data}")
    if not os.path.exists(args.train_data):
        logger.error(f"❌ File not found: {args.train_data}")
        sys.exit(1)
    
    try:
        X_train = pd.read_csv(args.train_data)
    except Exception as e:
        logger.error(f"❌ Failed to load training data: {e}")
        sys.exit(1)
    
    # Remove label column if present
    label_candidates = ['label', 'Label', 'class', 'Class', 'attack', 'Attack', 'y']
    for col in label_candidates:
        if col in X_train.columns:
            logger.info(f"Dropping label column: {col}")
            X_train = X_train.drop(columns=[col])
            break
    
    # Build pipeline
    pipeline, feature_names = build_correct_pipeline(X_train, args.output)
    
    # Optional: validate on test data if provided
    if args.test_data:
        logger.info(f"\nLoading test data from: {args.test_data}")
        if os.path.exists(args.test_data):
            try:
                X_test = pd.read_csv(args.test_data)
                # Remove label if present
                for col in label_candidates:
                    if col in X_test.columns:
                        X_test = X_test.drop(columns=[col])
                        break
                
                validate_pipeline(pipeline, X_train, X_test)
            except Exception as e:
                logger.error(f"❌ Failed to validate with test data: {e}")
        else:
            logger.warning(f"⚠️  Test data not found: {args.test_data}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ PIPELINE REBUILD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nPipeline saved to: {args.output}")
    logger.info(f"Features: {len(feature_names)}")
    logger.info("\nNext steps:")
    logger.info("  1. Retrain your RF/XGBoost models with this new pipeline")
    logger.info("  2. Test on holding-out test data")
    logger.info("  3. Verify predictions are no longer collapsed to BENIGN")
    logger.info("\nSee PREPROCESSING_DATA_LEAKAGE_ANALYSIS.md for guidance\n")


if __name__ == '__main__':
    main()
