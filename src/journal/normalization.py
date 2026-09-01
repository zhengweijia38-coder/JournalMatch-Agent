"""Shared normalization rules for structured journal classification fields."""

import re


def normalize_ccf_rank(value: str | None) -> str | None:
    """Normalize recognized CCF A/B/C variants and preserve unknown text."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    compact = re.sub(r"\s+", "", text).upper()
    for rank in ("A", "B", "C"):
        if compact in {rank, f"{rank}类"}:
            return rank
    return text


def normalize_jcr_quartile(value: str | None) -> str | None:
    """Normalize JCR values such as q1 or 1 to the stored Q1 form."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    compact = re.sub(r"\s+", "", text).upper()
    match = re.fullmatch(r"Q?([1-4])", compact)
    return f"Q{match.group(1)}" if match else text


def normalize_cas_quartile(value: str | None) -> str | None:
    """Normalize CAS zone/Top variants to their comparable stored zone level."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    compact = re.sub(r"\s+", "", text).upper()
    match = re.fullmatch(r"([1-4])(?:区)?(?:-?TOP)?", compact)
    return f"{match.group(1)}区" if match else text
