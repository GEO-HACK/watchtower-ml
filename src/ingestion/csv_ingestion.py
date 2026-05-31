"""CSV ingestion layer for detection workflows.

This module reuses the existing CSV flow extraction behavior from capture_reader
so terminal and API execution paths stay aligned.
"""

import os
import tempfile
from typing import Optional, Tuple

import pandas as pd

from ingestion.capture_reader import csv_to_flow_features


DEBUG_SAMPLE = True
SAMPLE_SIZE = 100000


def load_flows_from_csv_path(csv_path: str) -> Tuple[pd.DataFrame, Optional[object]]:
    """Load flow rows from CSV using existing capture_reader logic.

    Returns a tuple of (flow_dataframe, optional_labels).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df, embedded_y = csv_to_flow_features(csv_path)
    return _apply_terminal_sampling(df), embedded_y


def load_flows_from_csv_upload(filename: str, content: bytes) -> Tuple[pd.DataFrame, Optional[object]]:
    """Load flow rows from uploaded CSV bytes through the same CSV parser path."""
    suffix = os.path.splitext(filename or "upload.csv")[1] or ".csv"
    with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        temp_path = tmp.name

    try:
        return load_flows_from_csv_path(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _apply_terminal_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror terminal workflow sampling behavior in run_detector.py."""
    if DEBUG_SAMPLE and len(df) > SAMPLE_SIZE:
        return df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    return df
