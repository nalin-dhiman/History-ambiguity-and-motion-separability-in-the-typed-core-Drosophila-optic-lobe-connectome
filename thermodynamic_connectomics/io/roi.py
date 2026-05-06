"""ROI parsing helpers for neuPrint optic-lobe exports."""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

MAJOR_ROIS = ("LO(R)", "ME(R)", "LOP(R)", "OL(R)")
SPECIFIC_ROIS = ("LO(R)", "ME(R)", "LOP(R)")


def _score_roi(payload: dict) -> float:
    score = 0.0
    for key in ("post", "pre", "synweight", "downstream", "upstream"):
        value = payload.get(key, 0)
        if isinstance(value, (int, float)):
            score += float(value)
    return score


@lru_cache(maxsize=200_000)
def major_roi_from_info(roi_info: str | None) -> str:
    """Assign one major optic-lobe ROI from a neuPrint roiInfo JSON string."""

    if roi_info is None or pd.isna(roi_info):
        return "other"
    if not isinstance(roi_info, str) or not roi_info:
        return "other"
    try:
        data = json.loads(roi_info)
    except (TypeError, ValueError, json.JSONDecodeError):
        return "other"
    if not isinstance(data, dict):
        return "other"

    scores = {}
    for roi in SPECIFIC_ROIS:
        payload = data.get(roi)
        if isinstance(payload, dict):
            scores[roi] = _score_roi(payload)
    if scores:
        return max(scores.items(), key=lambda kv: (kv[1], -SPECIFIC_ROIS.index(kv[0])))[0]
    payload = data.get("OL(R)")
    if isinstance(payload, dict) and _score_roi(payload) > 0:
        return "OL(R)"
    return "other"


def assign_major_roi_series(series: pd.Series) -> pd.Series:
    """Vectorized wrapper around cached major ROI assignment."""

    return series.map(major_roi_from_info).astype("string")
