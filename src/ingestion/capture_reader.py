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

FLOW_ID_ALIASES = {
    'src_ip': ['src_ip', 'source_ip', 'source', 'client_ip', 'ip_src'],
    'dst_ip': ['dst_ip', 'destination_ip', 'destination', 'server_ip', 'ip_dst'],
    'src_port': ['src_port', 'source_port', 'sport'],
    'dst_port': ['dst_port', 'destination_port', 'dport'],
    'protocol': ['protocol', 'proto', 'ip_proto'],
    'start_time': ['start_time', 'flow_start', 'start', 'timestamp', 'ts'],
    'end_time': ['end_time', 'flow_end', 'end', 'stop_time'],
}


def _first_present_value(row: pd.Series, candidates):
    for column_name in candidates:
        if column_name in row.index:
            value = row.get(column_name)
            if pd.notna(value) and str(value).strip() != '':
                return value
    return None


def _format_flow_id(row: pd.Series) -> str:
    src_ip = _first_present_value(row, FLOW_ID_ALIASES['src_ip'])
    dst_ip = _first_present_value(row, FLOW_ID_ALIASES['dst_ip'])
    src_port = _first_present_value(row, FLOW_ID_ALIASES['src_port'])
    dst_port = _first_present_value(row, FLOW_ID_ALIASES['dst_port'])
    protocol = _first_present_value(row, FLOW_ID_ALIASES['protocol'])
    start_time = _first_present_value(row, FLOW_ID_ALIASES['start_time'])
    end_time = _first_present_value(row, FLOW_ID_ALIASES['end_time'])

    base_id = f'{src_ip}:{src_port}->{dst_ip}:{dst_port}/proto={protocol}'
    if start_time is not None and end_time is not None:
        return f'{base_id}@{start_time}-{end_time}'
    return base_id


def _ensure_flow_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df_out = df.copy()
    if 'flow_id' in df_out.columns:
        df_out['flow_id'] = df_out['flow_id'].apply(
            lambda value: str(value) if pd.notna(value) and str(value).strip() != '' else None
        )
        return df_out

    if 'Flow ID' in df_out.columns:
        df_out = df_out.rename(columns={'Flow ID': 'flow_id'})
        df_out['flow_id'] = df_out['flow_id'].apply(
            lambda value: str(value) if pd.notna(value) and str(value).strip() != '' else None
        )
        return df_out

    if all(
        _first_present_value(df_out.iloc[0], FLOW_ID_ALIASES[key]) is not None
        for key in ('src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol')
    ):
        df_out['flow_id'] = df_out.apply(_format_flow_id, axis=1)
    return df_out


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
    return _ensure_flow_id_column(df)


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

    return _ensure_flow_id_column(df), y_test
