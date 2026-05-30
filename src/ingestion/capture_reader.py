"""Owns reading PCAP and CSV capture files and returning flow DataFrames with optional ground truth labels."""

import pandas as pd

from preprocessing.packet_capture import capture_from_pcap, get_packet_metadata
from preprocessing.flow_aggregator import FlowAggregator


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


def pcap_to_flow_features(pcap_path, max_packets=None, skip_packets=0):
    """Read a PCAP file and return a pandas.DataFrame of flow feature dicts."""
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
