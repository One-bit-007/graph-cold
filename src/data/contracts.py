"""Dataset contracts for the real-data acquisition gate.

Contracts describe files and schema needed before any experiment runner may use
the dataset. They are intentionally stricter than the loader: audit reports can
explain missing optional views, but no contract fabricates unsupported fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetContract:
    name: str
    root: str
    expected_files: list[str] | None = None
    label_column: str | None = None
    required_columns: list[str] = field(default_factory=list)
    required_any_columns: dict[str, list[str]] = field(default_factory=dict)
    min_samples: int = 0
    min_classes: int = 2
    expected_view_support: dict[str, bool] = field(default_factory=dict)
    source_verified: bool = True
    replacement_for: str | None = None
    replacement_name_must_be_reported: bool = False
    notes: str = ""


CICIDS2017_CONTRACT = DatasetContract(
    name="cicids2017",
    root="data/cicids2017",
    expected_files=[
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ],
    label_column="Label",
    required_any_columns={
        "src_ip": ["Source IP", "Src IP", "src_ip", "SourceIP"],
        "dst_ip": ["Destination IP", "Dst IP", "dst_ip", "DestinationIP"],
        "src_port": ["Source Port", "Src Port", "src_port"],
        "dst_port": ["Destination Port", "Dst Port", "dst_port"],
        "protocol": ["Protocol", "protocol"],
        "timestamp": ["Timestamp", "Flow Timestamp", "timestamp"],
    },
    min_samples=10000,
    min_classes=2,
    expected_view_support={
        "host": True,
        "ip": True,
        "temporal": True,
        "process": False,
        "threat_intel": False,
    },
)


MALTLS22_CONTRACT = DatasetContract(
    name="maltls22",
    root="data/maltls22",
    expected_files=None,
    label_column="label",
    required_any_columns={
        "flow_or_tls_features": ["tls", "TLS", "flow", "Flow", "packet", "Packet"],
        "timestamp": ["timestamp", "Timestamp", "time", "Time"],
    },
    min_samples=10000,
    min_classes=2,
    expected_view_support={
        "host": True,
        "ip": True,
        "temporal": True,
        "process": False,
        "threat_intel": False,
    },
    source_verified=False,
    notes="Conditional only: source/access is not verified locally.",
)


TLS_ALTERNATIVE_CONTRACT = DatasetContract(
    name="tls_alternative",
    root="data/tls_alternative",
    expected_files=None,
    label_column=None,
    required_any_columns={
        "label": ["label", "Label", "class", "Class", "family", "Family"],
        "tls_features": ["tls", "TLS", "sni", "SNI", "ja3", "JA3", "flow", "Flow"],
    },
    min_samples=10000,
    min_classes=2,
    expected_view_support={
        "host": True,
        "ip": True,
        "temporal": True,
        "process": False,
        "threat_intel": False,
    },
    source_verified=False,
    notes="Candidate template only; never participates in D5 by default.",
)


CESNET_TLS_YEAR22_CONTRACT = DatasetContract(
    name="cesnet_tls_year22",
    root="data/tls_alternative/cesnet_tls_year22",
    expected_files=None,
    label_column=None,
    required_any_columns={
        "label": ["label", "Label", "class", "Class", "service", "app", "target", "category"],
        "tls_or_flow_features": [
            "tls",
            "TLS",
            "sni",
            "SNI",
            "ja3",
            "JA3",
            "flow",
            "Flow",
            "packet",
            "Packet",
            "bytes",
            "duration",
        ],
        "timestamp": ["timestamp", "Timestamp", "time", "Time", "start_time", "date"],
    },
    min_samples=10000,
    min_classes=2,
    expected_view_support={
        "host": False,
        "ip": True,
        "temporal": True,
        "process": False,
        "threat_intel": False,
    },
    source_verified=True,
    replacement_for="maltls22",
    replacement_name_must_be_reported=True,
    notes="Verified replacement candidate; report by true name CESNET-TLS-Year22, not MALTLS-22.",
)


OPTC_CONTRACT = DatasetContract(
    name="optc",
    root="data/optc",
    expected_files=["events.csv"],
    label_column="label",
    required_columns=[
        "host_id",
        "process_id",
        "parent_process_id",
        "src_ip",
        "dst_ip",
        "timestamp",
        "event_type",
        "alert_type",
        "label",
        "risk_score",
    ],
    min_samples=1000,
    min_classes=2,
    expected_view_support={
        "host": True,
        "ip": True,
        "temporal": True,
        "process": True,
        "threat_intel": True,
    },
)


DATASET_CONTRACTS = {
    "cicids2017": CICIDS2017_CONTRACT,
    "maltls22": MALTLS22_CONTRACT,
    "cesnet_tls_year22": CESNET_TLS_YEAR22_CONTRACT,
    "optc": OPTC_CONTRACT,
}
