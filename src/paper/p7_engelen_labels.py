"""P1-4: Faithful port of Engelen et al. CNS2022 CICIDS-2017 original-version labelling rules.

Source (verbatim port, no rule changes):
  E:/graphcold-data/cicids2017_corrected/CNS2022_Code/Labelling/CICIDS2017_original_version_labelling.ipynb
  (github.com/GintsEngelen/CNS2022_Code)

The notebook accepts the ORIGINAL CICIDS-2017 CSVs as released by CIC (our WP-3 raw
copy schema) and re-labels flows row-by-row, so row-level alignment with the raw
copy is natural: output row i of file F corresponds to input row i of file F.

Deviations from the notebook (behaviour-preserving, documented):
  * only the columns needed by the rules are read (usecols) to bound memory;
  * prints are silenced; CSV writing is replaced by an in-memory/DataFrame API;
  * `floor(freq='T')` becomes `floor(freq='min')` (pandas>=3 removed the 'T'
    alias; semantics identical);
  * labels are additionally normalised to a canonical namespace for diffing.

Attempted-flow discipline (mandated by the dataset authors' site and the WP
prompt): any flow with Attempted Category != -1 is mapped to BENIGN and is NEVER
an ML class. `apply_attempted_discipline` hard-codes this and asserts on it.

Timestamp reconstruction (verbatim from the notebook, do NOT "fix"):
  parse '%d/%m/%Y %H:%M' -> if hour < 7 add 12h (authors used 12-hour clock but
  dropped AM/PM; traffic ran 9:00-17:00) -> add 3h (New Brunswick DST = UTC-3).
  Attack windows below are hard-coded unix nanoseconds, kept exactly as published.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Columns the rules touch (behaviour-preserving usecols subset).
RULE_COLUMNS = [
    "Flow ID",
    "Timestamp",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Label",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Duration",
    "Fwd Packet Length Max",
]

# Canonical label namespace (Engelen-style). Original-release labels are mapped
# onto the same namespace so that a row-level diff is meaningful.
_CANON_FROM_ORIGINAL = {
    "BENIGN": "BENIGN",
    "Bot": "Botnet",
    "DDoS": "DDoS",
    "DoS GoldenEye": "DoS GoldenEye",
    "DoS Hulk": "DoS Hulk",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "DoS slowloris": "DoS Slowloris",
    "FTP-Patator": "FTP-Patator",
    "Heartbleed": "Heartbleed",
    "Infiltration": "Infiltration",
    "PortScan": "Portscan",
    "SSH-Patator": "SSH-Patator",
    "Web Attack – Brute Force": "Web Attack - Brute Force",
    "Web Attack – Sql Injection": "Web Attack - SQL Injection",
    "Web Attack – XSS": "Web Attack - XSS",
    "Web Attack \x96 Brute Force": "Web Attack - Brute Force",
    "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
    "Web Attack \ufffd Sql Injection": "Web Attack - SQL Injection",
    "Web Attack \ufffd XSS": "Web Attack - XSS",
}


def canonicalize_original(label: str) -> str:
    """Map an original-release CICIDS-2017 label to the canonical namespace."""
    key = str(label).strip()
    if key in _CANON_FROM_ORIGINAL:
        return _CANON_FROM_ORIGINAL[key]
    # tolerate any residual dash/spacing variants
    norm = key.replace("–", "-").replace("\x96", "-").replace("\ufffd", "-")
    norm = "-".join(part.strip() for part in norm.split("-"))
    if norm in _CANON_FROM_ORIGINAL:
        return _CANON_FROM_ORIGINAL[norm]
    return norm


def apply_attempted_discipline(corrected: pd.Series, attempted: pd.Series) -> pd.Series:
    """Attempted Category != -1  ->  BENIGN. Hard-coded, asserted, never an ML class."""
    corrected = corrected.astype("string")
    attempted = pd.to_numeric(attempted)
    out = corrected.mask(attempted != -1, "BENIGN")
    # assertions: no Attempted label survives; every attempted row is BENIGN
    assert not out.str.contains("Attempted", na=False).any(), "Attempted label leaked into final labels"
    assert (out[attempted != -1] == "BENIGN").all(), "attempted flow not mapped to BENIGN"
    return out


# ---------------------------------------------------------------------------
# Verbatim port of the notebook helpers (cell 1)
# ---------------------------------------------------------------------------

def format_csv_for_labelling(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lstrip(" ")

    # Since CICIDS 2017 authors used 12-hour format but removed AM/PM, we need to reconstruct it
    # We do this based on the knowledge they collected traffic from 9:00 AM to 5:00 PM.
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M")
    except ValueError:
        # Our raw mirror carries a few rows with seconds ('%d/%m/%Y %H:%M:%S');
        # 'mixed' + dayfirst parses both shapes with identical semantics.
        warnings.warn("timestamp column has seconds-bearing rows; parsed with format='mixed', dayfirst=True", stacklevel=2)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=True)
    df["Timestamp"] = df["Timestamp"].apply(lambda x: x + pd.DateOffset(hours=12) if x.hour < 7 else x)

    # Convert to UTC from New Brunswick summer timezone (UTC-3)
    df["Timestamp"] = df["Timestamp"] + pd.DateOffset(hours=3)

    for column in df.columns:
        if column not in ["Flow ID", "Timestamp", "Source IP", "Destination IP", "Label"]:
            df[column] = pd.to_numeric(df[column])

    # Add attempted category column and initialise to -1
    df["Attempted Category"] = -1

    # CICIDS 2017 author-released version comes prelabelled. This makes sure previous labels don't interfere
    df["Label"] = "NeedManualLabel"

    return df


def label_flows(df, label, attack_start_time_nanoseconds, attack_end_time_nanoseconds, src_ip_list=None,
                dst_ip_list=None, src_port_list=None, dst_port_list=None, attempted_category=-1, additional_filters=[],
                also_flip_flow_direction=False, payload_filter=False):
    # Note that labelling happens inplace on the 'df' parameter, and so this function doesn't return anything

    # Create initial mask with all values set to True. Squeeze is necessary to remove second axis (with value 1)
    custom_mask = pd.DataFrame(True, index=df.index, columns=[df.columns[0]]).squeeze()

    # Need to round the start time down to the nearest minute because otherwise some flows at the start of the attack
    # are labelled as benign. Notebook used freq='T'; pandas>=3 removed that alias, 'min' is identical.
    attack_start_datetime = pd.to_datetime(attack_start_time_nanoseconds, unit="ns").floor(freq="min")
    attack_end_datetime = pd.to_datetime(attack_end_time_nanoseconds, unit="ns")

    custom_mask &= (df["Timestamp"] >= attack_start_datetime)
    custom_mask &= (df["Timestamp"] <= attack_end_datetime)

    if src_ip_list is not None:
        custom_mask &= (df["Source IP"].isin(src_ip_list))
    if dst_ip_list is not None:
        custom_mask &= (df["Destination IP"].isin(dst_ip_list))

    if src_port_list is not None:
        custom_mask &= (df["Source Port"].isin(src_port_list))
    if dst_port_list is not None:
        custom_mask &= (df["Destination Port"].isin(dst_port_list))

    if payload_filter:
        custom_mask &= (df["Total Length of Fwd Packets"] == 0)

    for filter in additional_filters:
        custom_mask &= filter

    df["Label"] = df["Label"].mask(custom_mask, label)
    df["Attempted Category"] = df["Attempted Category"].mask(custom_mask, attempted_category)

    if also_flip_flow_direction:
        if additional_filters:
            raise AttributeError("Cannot set also_flip_flow_direction to True when additional_filters is not empty")

        custom_mask = pd.DataFrame(True, index=df.index, columns=[df.columns[0]]).squeeze()

        custom_mask &= (df["Timestamp"] >= attack_start_datetime)
        custom_mask &= (df["Timestamp"] <= attack_end_datetime)

        if src_ip_list is not None:
            custom_mask &= (df["Destination IP"].isin(src_ip_list))
        if dst_ip_list is not None:
            custom_mask &= (df["Source IP"].isin(dst_ip_list))

        if src_port_list is not None:
            custom_mask &= (df["Destination Port"].isin(src_port_list))
        if dst_port_list is not None:
            custom_mask &= (df["Source Port"].isin(dst_port_list))

        if payload_filter:
            custom_mask &= (df["Total Length of Bwd Packets"] == 0)

        for filter in additional_filters:
            custom_mask &= filter

        df["Label"] = df["Label"].mask(custom_mask, label)
        df["Attempted Category"] = df["Attempted Category"].mask(custom_mask, attempted_category)


def label_rest_as_benign(df: pd.DataFrame) -> pd.DataFrame:
    df["Label"] = df["Label"].mask(df["Label"] == "NeedManualLabel", "BENIGN")

    # Relabel artefact flows with [Flow Id] = '8.0.6.4-8.6.0.1-0-0-0' to label = 0
    df["Label"] = df["Label"].mask(df["Flow ID"] == "8.0.6.4-8.6.0.1-0-0-0", "BENIGN")
    return df


# ---------------------------------------------------------------------------
# Per-file rule applications (verbatim port of notebook cells 2-5)
# ---------------------------------------------------------------------------

def label_tuesday(df: pd.DataFrame) -> pd.DataFrame:
    # FTP-PATATOR
    label_flows(df, "FTP-Patator", 1499170672838272000, 1499174416931403000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[21], also_flip_flow_direction=True)

    label_flows(df, "FTP-Patator - Attempted", 1499170672838272000, 1499174416931403000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[21], payload_filter=True, also_flip_flow_direction=True, attempted_category=0)

    label_flows(df, "FTP-Patator - Attempted", 1499170672838272000, 1499174416931403000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[21], src_port_list=[52108], attempted_category=2)

    label_flows(df, "FTP-Patator - Attempted", 1499170672838272000, 1499174416931403000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[21], additional_filters=[
                    (df["Source Port"] != 52108) & (df["Total Length of Bwd Packets"] == 0) &
                    (df["Total Length of Fwd Packets"] > 0)
                ], attempted_category=3)

    label_flows(df, "FTP-Patator - Attempted", 1499170672838272000, 1499174416931403000, ["192.168.10.50"],
                ["172.16.0.1"], src_port_list=[21], additional_filters=[
                    (df["Destination Port"] != 52108) & (df["Total Length of Fwd Packets"] == 0) &
                    (df["Total Length of Bwd Packets"] > 0)
                ], attempted_category=3)

    # SSH-Patator
    label_flows(df, "SSH-Patator", 1499188141049616000, 1499195059018486000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[22], also_flip_flow_direction=True)

    label_flows(df, "SSH-Patator - Attempted", 1499188141049616000, 1499195059018486000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[22], payload_filter=True, also_flip_flow_direction=True, attempted_category=0)

    label_flows(df, "SSH-Patator - Attempted", 1499188141049616000, 1499195059018486000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[22], additional_filters=[
                    (df["Total Length of Fwd Packets"] <= 32) & (df["Total Length of Bwd Packets"] == 0)
                ], attempted_category=3)

    return label_rest_as_benign(df)


def label_wednesday(df: pd.DataFrame) -> pd.DataFrame:
    # DoS Slowloris
    label_flows(df, "DoS Slowloris - Attempted", 1499258926211817000, 1499258927000000000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], attempted_category=5)

    label_flows(df, "DoS Slowloris", 1499258934539220000, 1499260278500956000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], additional_filters=[
            ~(df["Source Port"].isin([33358, 33360, 33362, 54114]))
        ])

    label_flows(df, "DoS Slowloris", 1499258934539220000, 1499260278500956000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], additional_filters=[
            ~(df["Destination Port"].isin([33358, 33360, 33362, 54114]))
        ])

    label_flows(df, "DoS Slowloris - Attempted", 1499258934539220000, 1499260278500956000, ["172.16.0.1"],
                ["192.168.10.50"], src_port_list=[33358, 33360, 33362], dst_port_list=[80], attempted_category=2)

    label_flows(df, "DoS Slowloris - Attempted", 1499258934539220000, 1499260278500956000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], attempted_category=0, payload_filter=True, additional_filters=[
            ~(df["Source Port"].isin([33358, 33360, 33362, 54114]))])

    label_flows(df, "DoS Slowloris - Attempted", 1499258934539220000, 1499260278500956000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=0, additional_filters=[
            ~(df["Destination Port"].isin([33358, 33360, 33362, 54114])) & (df["Total Length of Bwd Packets"] == 0)
        ])

    label_flows(df, "DoS Slowloris - Attempted", 1499258934539220000, 1499260278500956000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=6, additional_filters=[
            ~(df["Destination Port"].isin([33358, 33360, 33362, 54114])) & (df["Total Length of Bwd Packets"] == 0)
            & (df["Flow Duration"] >= 199800)
        ])

    label_flows(df, "DoS Slowloris - Attempted", 1499258934539220000, 1499260278500956000, ["172.16.0.1"],
                ["192.168.10.50"], src_port_list=[54114], dst_port_list=[80], attempted_category=4)

    # DoS Slowhttptest
    label_flows(df, "DoS Slowhttptest", 1499260537936810000, 1499261869331517000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], additional_filters=[
            ~(df["Source Port"].isin([33372]))
        ])

    label_flows(df, "DoS Slowhttptest", 1499260537936810000, 1499261869331517000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], additional_filters=[
            ~(df["Destination Port"].isin([33372]))
        ])

    label_flows(df, "DoS Slowhttptest - Attempted", 1499260537936810000, 1499261869331517000, ["172.16.0.1"],
                ["192.168.10.50"], src_port_list=[33372], dst_port_list=[80], attempted_category=2)

    label_flows(df, "DoS Slowhttptest - Attempted", 1499260537936810000, 1499261869331517000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], attempted_category=0, payload_filter=True, additional_filters=[
            ~(df["Source Port"].isin([33372, 37670]))])

    label_flows(df, "DoS Slowhttptest - Attempted", 1499260537936810000, 1499261869331517000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=0, additional_filters=[
            ~(df["Destination Port"].isin([33372, 37670])) & (df["Total Length of Bwd Packets"] == 0)
        ])

    label_flows(df, "DoS Slowhttptest - Attempted", 1499260537936810000, 1499261869331517000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], attempted_category=6, additional_filters=[
            ~(df["Source Port"].isin([33372])) & (df["Total Length of Fwd Packets"] == 0) &
            (df["Flow Duration"] >= 199984) & (df["Total Backward Packets"] == 0)
        ])

    label_flows(df, "DoS Slowhttptest - Attempted", 1499260537936810000, 1499261869331517000, ["172.16.0.1"],
                ["192.168.10.50"], src_port_list=[37670], dst_port_list=[80], attempted_category=4)

    # DoS Hulk
    label_flows(df, "DoS Hulk", 1499262203194704000, 1499262299999999999, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], additional_filters=[
            ~(df["Source Port"].isin([48678, 43664]))
        ])

    label_flows(df, "DoS Hulk", 1499262203194704000, 1499262299999999999,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], additional_filters=[
            ~(df["Destination Port"].isin([48678, 43664]))
        ])

    label_flows(df, "DoS Hulk", 1499262300000000000, 1499263641326171000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], also_flip_flow_direction=True)

    label_flows(df, "DoS Hulk - Attempted", 1499262203194704000, 1499262299999999999, ["172.16.0.1"],
                ["192.168.10.50"], src_port_list=[48678, 43664], dst_port_list=[80], attempted_category=4)

    label_flows(df, "DoS Hulk - Attempted", 1499262203194704000, 1499263641326171000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=0, payload_filter=True, additional_filters=[
            ~(df["Source Port"].isin([48678, 43664]))])

    label_flows(df, "DoS Hulk - Attempted", 1499262203194704000, 1499263641326171000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=0, additional_filters=[
            ~(df["Destination Port"].isin([48678, 43664])) & (df["Total Length of Bwd Packets"] == 0)
        ])

    label_flows(df, "DoS Hulk - Attempted", 1499262203194704000, 1499263641326171000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], attempted_category=3, additional_filters=[
            ~(df["Source Port"].isin([48678, 43664])) & (df["Total Length of Fwd Packets"] > 0)
            & (df["Total Length of Fwd Packets"] < 282)
        ])

    label_flows(df, "DoS Hulk - Attempted", 1499262203194704000, 1499263641326171000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=3, additional_filters=[
            ~(df["Destination Port"].isin([48678, 43664])) & (df["Total Length of Bwd Packets"] > 0)
            & (df["Total Length of Bwd Packets"] < 282)
        ])

    # DoS GoldenEye
    label_flows(df, "DoS GoldenEye", 1499263803231753000, 1499264408915718000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], also_flip_flow_direction=True)

    label_flows(df, "DoS GoldenEye - Attempted", 1499263803231753000, 1499264408915718000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], attempted_category=0, payload_filter=True, also_flip_flow_direction=True)

    # Heartbleed
    label_flows(df, "Heartbleed", 1499278335650811000, 1499279563294455000, ["172.16.0.1"],
                ["192.168.10.51"], dst_port_list=[444], src_port_list=[45022], also_flip_flow_direction=True)

    label_flows(df, "Heartbleed - Attempted", 1499278335650811000, 1499279563294455000, ["172.16.0.1"],
                ["192.168.10.51"], dst_port_list=[444], src_port_list=[45022], attempted_category=0, payload_filter=True, also_flip_flow_direction=True)

    return label_rest_as_benign(df)


def label_thursday_web_attacks(df: pd.DataFrame) -> pd.DataFrame:
    # Web Attack - Brute Force
    label_flows(df, "Web Attack - Brute Force - Attempted", 1499343354880049000, 1499343531179279000,
                ["172.16.0.1"], ["192.168.10.50"], dst_port_list=[80], attempted_category=2, also_flip_flow_direction=True)

    label_flows(df, "Web Attack - Brute Force", 1499343567660566000, 1499346011622209000,
                ["172.16.0.1"], ["192.168.10.50"], dst_port_list=[80], additional_filters=[
                (df["Total Fwd Packets"] > 20) | (df["Source Port"] == 44464)
            ])

    label_flows(df, "Web Attack - Brute Force", 1499343567660566000, 1499346011622209000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], additional_filters=[
                (df["Total Backward Packets"] > 20) | (df["Destination Port"] == 44464)
            ])

    label_flows(df, "Web Attack - Brute Force - Attempted", 1499343567660566000, 1499346011622209000,
                ["172.16.0.1"], ["192.168.10.50"], dst_port_list=[80], payload_filter=True, attempted_category=0,
                additional_filters=[
                ~((df["Total Fwd Packets"] > 20) | (df["Source Port"] == 44464))])

    label_flows(df, "Web Attack - Brute Force - Attempted", 1499343567660566000, 1499346011622209000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=0,
                additional_filters=[
                ~((df["Total Backward Packets"] > 20) | (df["Destination Port"] == 44464))
                & (df["Total Length of Bwd Packets"] == 0)
            ])

    label_flows(df, "Web Attack - Brute Force - Attempted", 1499343567660566000, 1499346011622209000,
                ["172.16.0.1"], ["192.168.10.50"], dst_port_list=[80], attempted_category=4,
                additional_filters=[
                (df["Total Length of Fwd Packets"] > 0) & ~(df["Source Port"] == 44464) &
                (df["Total Fwd Packets"] == 4) & (df["Total Backward Packets"] == 4)
            ])

    label_flows(df, "Web Attack - Brute Force - Attempted", 1499343567660566000, 1499346011622209000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=4,
                additional_filters=[
                (df["Total Length of Bwd Packets"] > 0) & ~(df["Destination Port"] == 44464) &
                (df["Total Backward Packets"] == 4) & (df["Total Fwd Packets"] == 4)
            ])

    # Web Attack - XSS
    label_flows(df, "Web Attack - XSS", 1499346935283859000, 1499348121341704000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], additional_filters=[
                ~(df["Source Port"].isin([36180, 36182, 36184, 36186, 36188, 36190])) &
                (df["Total Fwd Packets"] >= 150)
            ])

    label_flows(df, "Web Attack - XSS", 1499346935283859000, 1499348121341704000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], additional_filters=[
                ~(df["Destination Port"].isin([36180, 36182, 36184, 36186, 36188, 36190])) &
                (df["Total Backward Packets"] >= 150)
            ])

    label_flows(df, "Web Attack - XSS - Attempted", 1499346935283859000, 1499348121341704000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], attempted_category=0, payload_filter=True, additional_filters=[
                ~(df["Source Port"].isin([36180, 36182, 36184, 36186, 36188, 36190]))])

    label_flows(df, "Web Attack - XSS - Attempted", 1499346935283859000, 1499348121341704000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=0,
                additional_filters=[
                ~(df["Destination Port"].isin([36180, 36182, 36184, 36186, 36188, 36190])) &
                (df["Total Length of Bwd Packets"] == 0)
            ])

    label_flows(df, "Web Attack - XSS - Attempted", 1499346935283859000, 1499348121341704000, ["172.16.0.1"],
                ["192.168.10.50"], dst_port_list=[80], attempted_category=2, additional_filters=[
                ~(df["Source Port"].isin([36180, 36182, 36184, 36186, 36188, 36190])) &
                (df["Total Length of Fwd Packets"] > 0) & (df["Total Fwd Packets"] < 150)
            ])

    label_flows(df, "Web Attack - XSS - Attempted", 1499346935283859000, 1499348121341704000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=2,
                additional_filters=[
                ~(df["Destination Port"].isin([36180, 36182, 36184, 36186, 36188, 36190])) &
                (df["Total Length of Bwd Packets"] > 0) & (df["Total Backward Packets"] < 150)
            ])

    # Web Attack - SQL Injection
    label_flows(df, "Web Attack - SQL Injection - Attempted", 1499348127852814000, 1499348145720612000,
                ["172.16.0.1"], ["192.168.10.50"], dst_port_list=[80], attempted_category=2,
                additional_filters=[
                df["Source Port"].isin([36180, 36182, 36184, 36186, 36188])
            ])

    label_flows(df, "Web Attack - SQL Injection - Attempted", 1499348127852814000, 1499348145720612000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80], attempted_category=2,
                additional_filters=[
                df["Destination Port"].isin([36180, 36182, 36184, 36186, 36188])
            ])

    label_flows(df, "Web Attack - SQL Injection", 1499348145732950000, 1499348575320284000,
                ["172.16.0.1"], ["192.168.10.50"], dst_port_list=[80],
                additional_filters=[
                ~(df["Source Port"].isin([36180, 36182, 36184, 36186, 36188]))
            ])

    label_flows(df, "Web Attack - SQL Injection", 1499348145732950000, 1499348575320284000,
                ["192.168.10.50"], ["172.16.0.1"], src_port_list=[80],
                additional_filters=[
                ~(df["Destination Port"].isin([36180, 36182, 36184, 36186, 36188]))
            ])

    label_flows(df, "Web Attack - SQL Injection - Attempted", 1499348127852814000, 1499348145720612000,
                ["172.16.0.1"], ["192.168.10.50"], dst_port_list=[80], attempted_category=0,
                payload_filter=True, also_flip_flow_direction=True)

    return label_rest_as_benign(df)


def label_thursday_infiltration(df: pd.DataFrame) -> pd.DataFrame:
    # 5.1 Dropbox Download
    label_flows(df, "Infiltration", 1499361542547210000, 1499366769364731000, ["192.168.10.8"], ["205.174.165.73"],
                also_flip_flow_direction=True)

    label_flows(df, "Infiltration - Attempted", 1499361542547210000, 1499366769364731000, ["192.168.10.8"],
                ["205.174.165.73"], attempted_category=0, payload_filter=True, also_flip_flow_direction=True)

    label_flows(df, "Infiltration - Attempted", 1499361228830533000, 1499361301251276000, ["192.168.10.9"],
                ["205.174.165.73"], attempted_category=2, also_flip_flow_direction=True)

    # 5.2 Cooldisk Mac
    label_flows(df, "Infiltration", 1499363616453990000, 1499371339347892000, ["192.168.10.25"], ["205.174.165.73"],
                also_flip_flow_direction=True)

    label_flows(df, "Infiltration - Attempted", 1499363616453990000, 1499371339347892000, ["192.168.10.25"],
                ["205.174.165.73"], attempted_category=0, payload_filter=True, also_flip_flow_direction=True)

    # 5.3 NMAP + Portscan
    label_flows(df, "Infiltration - Portscan", 1499360400000000000, 1499360460000000000, ["172.16.0.1"],
                ["192.168.10.51"], additional_filters=[
            (df["Source Port"] == 50122)
        ])

    label_flows(df, "Infiltration - Portscan", 1499362410884008000, 1499362444285175000, ["192.168.10.8"],
                ["192.168.10.5"])

    label_flows(df, "Infiltration - Portscan", 1499364314425162000, 1499366764331875000, ["192.168.10.8"],
                ["192.168.10.5", "192.168.10.9", "192.168.10.12", "192.168.10.14", "192.168.10.15", "192.168.10.16",
                 "192.168.10.17", "192.168.10.19", "192.168.10.25", "192.168.10.50", "192.168.10.51"], additional_filters=[
            ~((df["Fwd Packet Length Max"] == 408) & (df["Destination IP"] == "192.168.10.50")) &
            ~((df["Total Length of Fwd Packets"].isin([176, 20514])) & (df["Destination IP"] == "192.168.10.50"))
        ])

    return label_rest_as_benign(df)


def _label_friday_botnet(df: pd.DataFrame) -> None:
    """Botnet rules are applied inside every Friday file (verbatim from notebook)."""
    label_flows(df, "Botnet", 1499432653990571000, 1499436122903736000, ["192.168.10.15", "192.168.10.9",
                "192.168.10.14", "192.168.10.5", "192.168.10.8"], ["205.174.165.73"], also_flip_flow_direction=True)

    label_flows(df, "Botnet - Attempted", 1499432653990571000, 1499436122903736000, ["192.168.10.15", "192.168.10.9",
                "192.168.10.14", "192.168.10.5", "192.168.10.8"], ["205.174.165.73"], attempted_category=0,
                payload_filter=True, also_flip_flow_direction=True)

    label_flows(df, "Botnet - Attempted", 1499436180000000000, 1499457684606663000, ["192.168.10.15", "192.168.10.9",
                "192.168.10.14", "192.168.10.5", "192.168.10.8"], ["205.174.165.73"], attempted_category=1, also_flip_flow_direction=True)


def label_friday_portscan(df: pd.DataFrame) -> pd.DataFrame:
    label_flows(df, "Portscan", 1499446532117090000, 1499447948582083000, ["172.16.0.1"], ["192.168.10.50"],
                also_flip_flow_direction=True)

    label_flows(df, "Portscan", 1499449860000000000, 1499449919000000000, ["172.16.0.1"], ["192.168.10.50"],
                additional_filters=[
                ~(df["Source Port"].isin([0, 35952, 35954, 35956, 35958]))
            ])

    label_flows(df, "Portscan", 1499449920000000000, 1499451841699238000, ["172.16.0.1"], ["192.168.10.50"])

    _label_friday_botnet(df)
    return label_rest_as_benign(df)


def label_friday_morning(df: pd.DataFrame) -> pd.DataFrame:
    _label_friday_botnet(df)
    return label_rest_as_benign(df)


def label_friday_ddos(df: pd.DataFrame) -> pd.DataFrame:
    label_flows(df, "DDoS", 1499453791796937000, 1499454972216560000, ["172.16.0.1"], ["192.168.10.50"],
                also_flip_flow_direction=True)

    label_flows(df, "DDoS - Attempted", 1499453791796937000, 1499454972216560000, ["172.16.0.1"], ["192.168.10.50"],
                attempted_category=0, payload_filter=True, also_flip_flow_direction=True)

    label_flows(df, "DDoS - Attempted", 1499453791796937000, 1499454972216560000, ["192.168.10.50"], ["172.16.0.1"],
                attempted_category=0, additional_filters=[
            (df["Total Length of Bwd Packets"] == 0)
        ])

    _label_friday_botnet(df)
    return label_rest_as_benign(df)


def label_monday(df: pd.DataFrame) -> pd.DataFrame:
    """Monday carries no attack rules in the notebook (all-benign capture)."""
    return label_rest_as_benign(df)


FILE_RULES = {
    "Monday-WorkingHours.pcap_ISCX.csv": label_monday,
    "Tuesday-WorkingHours.pcap_ISCX.csv": label_tuesday,
    "Wednesday-workingHours.pcap_ISCX.csv": label_wednesday,
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": label_thursday_web_attacks,
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": label_thursday_infiltration,
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": label_friday_morning,
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": label_friday_portscan,
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": label_friday_ddos,
}


def _read_raw_file(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0, low_memory=False).columns
    actual = [c for c in header if c.strip() in RULE_COLUMNS]
    try:
        df = pd.read_csv(path, usecols=actual, low_memory=False)
    except UnicodeDecodeError:
        warnings.warn(f"UTF-8 decode failed for {path.name}; fell back to cp1252", stacklevel=2)
        df = pd.read_csv(path, usecols=actual, low_memory=False, encoding="cp1252")
    df.columns = df.columns.str.lstrip(" ")
    return df


def derive_corrected_labels(raw_dir: str | Path) -> pd.DataFrame:
    """Apply the ported Engelen rules to every raw file.

    Returns a long DataFrame aligned 1:1 with the raw files' row order:
    [source_file, row_in_file, orig_label, corrected_rule_label, attempted_category,
     corrected_label, orig_canon, corrected_canon]
    where corrected_label already has the Attempted->BENIGN discipline applied.
    """
    raw_dir = Path(raw_dir)
    frames = []
    for file in sorted(raw_dir.glob("*.csv")):
        if file.name not in FILE_RULES:
            raise ValueError(f"No ported rule set for raw file {file.name}")
        df = _read_raw_file(file)
        orig_label = df["Label"].astype("string").str.strip()
        df = format_csv_for_labelling(df)
        df = FILE_RULES[file.name](df)
        corrected_rule = df["Label"].astype("string")
        attempted = pd.to_numeric(df["Attempted Category"])
        corrected_final = apply_attempted_discipline(corrected_rule, attempted)
        frames.append(pd.DataFrame(
            {
                "source_file": file.name,
                "row_in_file": np.arange(len(df), dtype=np.int64),
                "orig_label": orig_label.to_numpy(),
                "corrected_rule_label": corrected_rule.to_numpy(),
                "attempted_category": attempted.to_numpy(dtype=np.int64),
                "corrected_label": corrected_final.to_numpy(),
                "orig_canon": [canonicalize_original(v) for v in orig_label],
                "corrected_canon": corrected_final.to_numpy(),
            }
        ))
    return pd.concat(frames, ignore_index=True)


def diff_summary(labels: pd.DataFrame) -> dict:
    """Characterise the real mislabel set: original vs corrected canonical labels."""
    valid = labels["orig_label"].notna()
    df = labels[valid].copy()
    diff = df["orig_canon"] != df["corrected_canon"]
    by_pair = (
        df[diff].groupby(["orig_canon", "corrected_canon"]).size().sort_values(ascending=False)
    )
    return {
        "n_rows_total": int(len(labels)),
        "n_rows_valid_label": int(len(df)),
        "n_differ": int(diff.sum()),
        "diff_share_valid": float(diff.mean()),
        "n_attempted_to_benign": int((labels["attempted_category"] != -1).sum()),
        "by_orig_corrected_pair": {f"{o} -> {c}": int(n) for (o, c), n in by_pair.items()},
        "corrected_class_counts": df["corrected_canon"].value_counts().to_dict(),
        "original_class_counts": df["orig_canon"].value_counts().to_dict(),
    }


def main() -> None:
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser(description="Derive Engelen-corrected labels for the raw CICIDS-2017 copy")
    parser.add_argument("--raw-dir", default="E:/graphcold-data/cicids2017_raw/TrafficLabelling/")
    parser.add_argument("--out", default="results/p7_corrected_labels.csv.gz")
    parser.add_argument("--summary", default="reports/p7_engelen_diff_summary.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    labels = derive_corrected_labels(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False, compression="gzip")
    summary = diff_summary(labels)
    summary["derive_seconds"] = round(time.perf_counter() - t0, 1)
    summary["source_notebook"] = (
        "CNS2022_Code/Labelling/CICIDS2017_original_version_labelling.ipynb "
        "(github.com/GintsEngelen/CNS2022_Code)"
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict) or k == "by_orig_corrected_pair"}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
