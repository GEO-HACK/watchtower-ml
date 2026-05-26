#!/usr/bin/env python3
"""
Rebuild preprocessing pipeline to resolve sklearn version compatibility issues.

Use this script when you get "NotFittedError: Pipeline is not fitted yet"
during inference. This usually indicates that the pipeline.pkl was saved
with a different sklearn version or wasn't properly fitted.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing.custom_transformers import build_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def rebuild_pipeline():
    """Rebuild the preprocessing pipeline using available data and extracting feature names."""
    project_root = Path(__file__).parent
    data_dir = project_root / 'src' / 'data'
    models_dir = project_root / 'src' / 'models'
    
    # Paths
    X_train_path = data_dir / 'X_train.npy'
    X_test_path = data_dir / 'X_test.npy'
    pipeline_path = models_dir / 'preprocessing_pipeline.pkl'
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Models directory: {models_dir}")
    
    # Step 1: Try to extract feature names from existing pipeline
    logger.info("\nStep 1: Extracting feature names from existing pipeline...")
    feature_names = None
    if pipeline_path.exists():
        try:
            old_pipeline = joblib.load(pipeline_path)
            if hasattr(old_pipeline, 'named_steps'):
                feature_aligner = old_pipeline.named_steps.get('feature_aligner')
                if feature_aligner and hasattr(feature_aligner, 'feature_names'):
                    feature_names = list(feature_aligner.feature_names)
                    logger.info(f"✓ Extracted {len(feature_names)} feature names from existing pipeline")
        except Exception as e:
            logger.warning(f"Could not extract from existing pipeline: {e}")
    
    if feature_names is None:
        logger.error("✗ Could not extract feature names from pipeline")
        logger.error("This pipeline may be too corrupted to recover.")
        return False
    
    # Step 2: Find training/test data
    logger.info("\nStep 2: Locating training or test data...")
    X_data = None
    data_source = None
    
    if X_train_path.exists():
        logger.info(f"Found training data: {X_train_path}")
        X_data = np.load(X_train_path)
        data_source = 'X_train.npy'
    elif X_test_path.exists():
        logger.info(f"Training data not found, using test data: {X_test_path}")
        X_data = np.load(X_test_path)
        data_source = 'X_test.npy'
    else:
        logger.error("✗ Neither training nor test data found")
        return False
    
    logger.info(f"✓ Loaded {data_source}: shape {X_data.shape}")
    
    # Step 3: Rebuild pipeline
    logger.info("\nStep 3: Rebuilding preprocessing pipeline...")
    try:
        pipeline = build_preprocessing_pipeline(feature_names)
        logger.info(f"✓ Pipeline created")
    except Exception as e:
        logger.error(f"✗ Failed to build pipeline: {e}")
        return False
    
    # Step 4: Fit pipeline
    logger.info(f"\nStep 4: Fitting pipeline on {len(X_data)} samples...")
    try:
        X_df = pd.DataFrame(X_data, columns=feature_names)
        pipeline.fit(X_df)
        logger.info("✓ Pipeline fitted successfully")
    except Exception as e:
        logger.error(f"✗ Failed to fit pipeline: {e}")
        return False
    
    # Step 5: Verify all steps are fitted
    logger.info("\nStep 5: Verifying pipeline steps...")
    if hasattr(pipeline, 'named_steps'):
        for step_name, step_obj in pipeline.named_steps.items():
            has_fitted = (
                hasattr(step_obj, 'n_features_in_') or
                hasattr(step_obj, 'statistics_') or
                hasattr(step_obj, 'scale_') or
                hasattr(step_obj, 'mean_')
            )
            status = "✓" if has_fitted else "✗"
            logger.info(f"  {status} {step_name:20s} - {type(step_obj).__name__}")
    
    # Step 6: Save pipeline with backup
    logger.info(f"\nStep 6: Saving rebuilt pipeline...")
    backup_path = pipeline_path.with_suffix('.pkl.backup')
    if pipeline_path.exists():
        logger.info(f"  Backing up existing pipeline to {backup_path}")
        import shutil
        shutil.copy(pipeline_path, backup_path)
    
    try:
        joblib.dump(pipeline, pipeline_path, compress=3)
        logger.info(f"✓ Pipeline saved to {pipeline_path}")
    except Exception as e:
        logger.error(f"✗ Failed to save pipeline: {e}")
        return False
    
    # Step 7: Test the saved pipeline
    logger.info(f"\nStep 7: Testing saved pipeline...")
    try:
        loaded_pipeline = joblib.load(pipeline_path)
        X_test_df = X_df.iloc[:min(100, len(X_df))]
        X_transformed = loaded_pipeline.transform(X_test_df)
        logger.info(f"✓ Test transform successful: {X_test_df.shape} → {X_transformed.shape}")
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        return False
    
    logger.info("\n" + "="*80)
    logger.info("SUCCESS: Pipeline rebuilt and saved successfully!")
    logger.info(f"  Pipeline:      {pipeline_path}")
    logger.info(f"  Backup:        {backup_path}")
    logger.info(f"  Feature count: {len(feature_names)}")
    logger.info("="*80)
    return True


def check_sklearn_version():
    """Check and report sklearn version info."""
    import sklearn
    logger.info(f"scikit-learn version: {sklearn.__version__}")


if __name__ == '__main__':
    logger.info("="*80)
    logger.info("PREPROCESSING PIPELINE REBUILD UTILITY")
    logger.info("="*80)
    
    check_sklearn_version()
    success = rebuild_pipeline()
    
    sys.exit(0 if success else 1)
