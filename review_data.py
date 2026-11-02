"""Dated data preparation with explicit historical-link limitations.

The current SEC ticker snapshot is available only in legacy_diagnostic mode.
It is not promoted to a historically verified security/issuer relationship.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAIN_FORMS = ["10-K", "10-Q", "10KSB", "10QSB", "10-K405", "10KSB40", "10-KT", "10-QT"]
COUNT_COLUMNS = ["N_Positive", "N_Negative", "N_Uncertainty", "N_Litigious", "N_StrongModal", "N_WeakModal", "N_Constraining"]
RATE_COLUMNS = ["positive_rate", "negative_rate", "uncertainty_rate", "litigious_rate", "strong_modal_rate", "weak_modal_rate", "constraining_rate"]


def prepare_return_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    """Preserve missing source outcomes and pair targets before universe filtering."""
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["formation_month"] = frame["date"].dt.to_period("M")
    if frame.duplicated(["PERMNO", "formation_month"]).any():
        raise ValueError("Duplicate PERMNO-calendar-month source records.")
    frame["source_return_code"] = frame["RET"].fillna("<blank>").astype(str)
    frame["RET"] = pd.to_numeric(frame["RET"], errors="coerce")
    frame["Ticker"] = frame["Ticker"].astype("string").str.strip().str.upper()
    target = frame[["PERMNO", "formation_month", "date", "RET", "source_return_code"]].rename(columns={
        "formation_month": "return_month", "date": "target_source_date", "RET": "Next_Month_Return",
        "source_return_code": "target_return_code",
    })
    frame["return_month"] = frame["formation_month"] + 1
    frame = frame.merge(target, on=["PERMNO", "return_month"], how="left", validate="many_to_one", indicator="_target_join")
    frame["target_status"] = np.where(frame["_target_join"].eq("left_only"), "absent_source_row",
        np.where(frame["Next_Month_Return"].notna(), "known", "source_return_missing_or_nonnumeric"))
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    return frame.drop(columns="_target_join")


def attach_identifiers(returns: pd.DataFrame, *, mode: str, sec_mapping=None, historical_links=None) -> pd.DataFrame:
    if mode == "legacy_diagnostic":
        if sec_mapping is None:
            raise ValueError("A frozen SEC snapshot is required for legacy diagnostics.")
        mapping = sec_mapping.rename(columns={"ticker": "Ticker", "cik_str": "CIK", "title": "CompanyName"}).copy()
        mapping["Ticker"] = mapping["Ticker"].astype("string").str.strip().str.upper()
        mapping["CIK"] = pd.to_numeric(mapping["CIK"], errors="raise").astype("int64")
        out = returns.merge(mapping[["Ticker", "CIK", "CompanyName"]], on="Ticker", how="left", validate="many_to_one")
        out["identifier_status"] = "unverified_current_ticker_snapshot"
        return out
    if mode != "verified_links":
        raise ValueError("Mode must be legacy_diagnostic or verified_links.")
    required = {"PERMNO", "CIK", "link_start", "link_end"}
    if historical_links is None or not required.issubset(historical_links.columns):
        raise ValueError("Verified mode requires dated PERMNO/CIK/link_start/link_end input; ticker or company-name guesses are not substituted.")
    links = historical_links[["PERMNO", "CIK", "link_start", "link_end"]].copy()
    links["PERMNO"] = pd.to_numeric(links["PERMNO"], errors="raise").astype("int64")
    links["CIK"] = pd.to_numeric(links["CIK"], errors="raise").astype("int64")
    links["link_start"] = pd.to_datetime(links["link_start"], errors="raise")
    links["link_end"] = pd.to_datetime(links["link_end"], errors="raise").fillna(pd.Timestamp("2262-04-11"))
    if links["link_start"].isna().any() or (links["link_start"] > links["link_end"]).any():
        raise ValueError("Invalid historical-link interval.")
    ordered = links.sort_values(["PERMNO", "link_start"])
    previous_end = ordered.groupby("PERMNO")["link_end"].shift()
    if ordered["link_start"].le(previous_end).any():
        raise ValueError("Overlapping historical-link intervals are ambiguous.")
    out = pd.merge_asof(
        returns.sort_values(["date", "PERMNO"]), links.sort_values(["link_start", "PERMNO"]),
        by="PERMNO", left_on="date", right_on="link_start", direction="backward",
    )
    out.loc[out["date"].gt(out["link_end"]), "CIK"] = np.nan
    out["identifier_status"] = "dated_link_input_supplied_not_independently_certified"
    return out


def attach_latest_filing(returns: pd.DataFrame, summaries: pd.DataFrame, *, lag_business_days: int = 1,
                         max_age_months: int = 3, baseline_formula: str = "published") -> pd.DataFrame:
    """Select one whole filing record; do not independently carry its fields.

    One business-day availability lag is a disclosed conservative convention
    when acceptance timestamps are missing. It is not an exchange calendar.
    Published formula: (P-N)/U + 1; guide variant: (P-N)/(U+1).
    """
    if (lag_business_days < 0 or int(lag_business_days) != lag_business_days
            or max_age_months < 0 or int(max_age_months) != max_age_months):
        raise ValueError("Availability lag and maximum age must be nonnegative integers.")
    if baseline_formula not in {"published", "guide"}:
        raise ValueError("Select published or guide baseline formula explicitly.")
    left = returns[returns["CIK"].notna() & returns["RET"].notna()].copy()
    left["CIK"] = pd.to_numeric(left["CIK"], errors="raise").astype("int64")
    right = summaries[summaries["FORM_TYPE"].str.strip().str.upper().isin(MAIN_FORMS)].copy()
    right["CIK"] = pd.to_numeric(right["CIK"], errors="raise").astype("int64")
    right["filing_date"] = pd.to_datetime(right["FILING_DATE"].astype(str), format="%Y%m%d", errors="raise")
    right["available_date"] = right["filing_date"] + pd.offsets.BDay(lag_business_days)
    right = right.sort_values(["available_date", "CIK", "ACC_NUM"], kind="stable").drop_duplicates(["CIK", "available_date"], keep="last")
    keep = ["CIK", "filing_date", "available_date", "ACC_NUM", "FORM_TYPE", "N_Words", "FFInd"] + COUNT_COLUMNS
    panel = pd.merge_asof(left.sort_values(["date", "CIK"]), right[keep].sort_values(["available_date", "CIK"]),
                          by="CIK", left_on="date", right_on="available_date", direction="backward")
    panel["filing_age_months"] = (panel["date"].dt.year - panel["filing_date"].dt.year) * 12 + panel["date"].dt.month - panel["filing_date"].dt.month
    panel["filing_usable"] = panel["filing_age_months"].between(0, max_age_months) & panel["N_Words"].gt(0)
    for count, rate in zip(COUNT_COLUMNS, RATE_COLUMNS):
        panel[rate] = (panel[count] / panel["N_Words"].where(panel["N_Words"] > 0)).where(panel["filing_usable"])
    spread = panel["N_Positive"] - panel["N_Negative"]
    published = spread / panel["N_Uncertainty"].replace(0, np.nan) + 1
    guide = spread / (panel["N_Uncertainty"] + 1)
    panel["Published_Baseline_Sentiment"] = published.where(panel["filing_usable"])
    panel["Guide_Baseline_Sentiment"] = guide.where(panel["filing_usable"])
    panel["Baseline_Sentiment"] = panel["Published_Baseline_Sentiment"] if baseline_formula == "published" else panel["Guide_Baseline_Sentiment"]
    z = []
    for rate in RATE_COLUMNS:
        mean = panel.groupby(["year", "month"])[rate].transform("mean")
        std = panel.groupby(["year", "month"])[rate].transform("std")
        z.append((panel[rate] - mean) / std.replace(0, np.nan))
    panel["Expanded_LM_Signal"] = z[0] - z[1] - z[2] - z[3] + z[4] - z[5] - z[6]
    if panel.loc[panel["filing_usable"], "available_date"].gt(panel.loc[panel["filing_usable"], "date"]).any():
        raise AssertionError("A filing was selected before the configured availability date.")
    return panel.sort_values(["PERMNO", "date"], kind="stable").reset_index(drop=True)
