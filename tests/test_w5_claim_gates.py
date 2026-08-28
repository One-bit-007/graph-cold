"""W5 claim gates: the revised manuscript must not regress on withdrawn claims.

Replaces the retired D6-D11 gates (tests/legacy/). Asserts the claim ledger's
banned phrasings stay absent and its required disclosures stay present in the
blind manuscript; runs the number audit as a subcheck.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

TEX_PATH = Path("paper/springer/graph_cold_sn_blind.tex")
TEX = TEX_PATH.read_text(encoding="utf-8")


def _hits(pattern: str) -> list[str]:
    return [m.group(0) for m in re.finditer(pattern, TEX, flags=re.IGNORECASE)]


BANNED = {
    # withdrawn performance claims
    "outperforms CoLD (positive claim)": r"outperform(s|ed|ing)?\s+CoLD\b(?![- ]aligned)",
    "superiority over official CoLD": r"(superior|better)\s+(to|than)\s+(official\s+)?CoLD",
    "evidence-rescue superiority": r"rescue (form|weighting) (is|was) superior",
    # banned vocabulary (F-6)
    "salvaged narrative": r"\bsalvag(ed|ing)\b(?!.*(rather than|not))",
    "operational/forensic evidence claim": r"(is|as|provides?|constitutes?)\s+(operational|forensic)\s+evidence",
    "rare-attack evidence": r"rare-attack evidence",
    # stale numbers from superseded protocols
    "stale P2f deltas": r"\+0\.0731|\+0\.0218|\+0\.0405",
    "stale proxy macro pair": r"0\.534.*?0\.688|0\.688.*?0\.534",
    # bare 'no improvement' instead of effect-ceiling language
    "bare no-improvement": r"no improvement\b",
}

REQUIRED = {
    "CoLD-aligned proxy naming": r"CoLD-aligned proxy",
    "non-comparability statement": r"not comparable in either\s+direction",
    "family B effect ceiling": r"\+0\.026",
    "family C withdrawal": r"withdraw the claim",
    "PA appendix": r"\\label\{app:amendments\}",
    "PA-1 disclosure": r"(?s)was \\emph\{not\}\s*pre-registered",
    "gap-compression direction": r"(?s)compress.*?soft",
    "CESNET semantics scope": r"service-class(ification)? stress[- ]test",
    "entity infeasibility finding": r"infeasible by construction",
    "UNSW collision disclosure": r"38\.4\\%",
    "Attempted-to-benign discipline source": r"labelling artefacts",
    "tuning discipline": r"no clean validation set",
}


@pytest.mark.parametrize("name,pattern", BANNED.items())
def test_banned_phrasing_absent(name: str, pattern: str) -> None:
    hits = _hits(pattern)
    assert not hits, f"banned ({name}): {hits[:3]}"


@pytest.mark.parametrize("name,pattern", REQUIRED.items())
def test_required_disclosure_present(name: str, pattern: str) -> None:
    assert re.search(pattern, TEX, flags=re.IGNORECASE), f"missing required disclosure: {name}"


def test_tables_are_generated_not_handwritten() -> None:
    for frag in Path("paper/springer/tables_w5").glob("tab_*.tex"):
        head = frag.read_text(encoding="utf-8").splitlines()[0]
        assert "AUTO-GENERATED" in head, f"{frag.name} lost its generation banner"


def test_number_audit_passes() -> None:
    proc = subprocess.run([sys.executable, "-m", "src.paper.w5_number_audit"],
                          capture_output=True, text=True, cwd=str(Path.cwd()))
    assert proc.returncode == 0, f"number audit failed:\n{proc.stdout[-2000:]}"
