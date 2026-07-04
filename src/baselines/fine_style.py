"""FINE-style baseline audit hook.

FINE is intentionally not emitted by D5.5 unless a faithful embedding-filtering
implementation is added and smoke-passes on both real datasets. This module
keeps the exclusion explicit so the experiment runner never fabricates FINE
numbers.
"""
from __future__ import annotations


def exclusion_reason() -> str:
    return (
        "excluded: faithful FINE requires a validated embedding outlier protocol; "
        "D5.5 keeps it out instead of reporting approximate or fake rows"
    )
