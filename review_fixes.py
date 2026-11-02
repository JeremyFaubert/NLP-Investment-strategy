"""Calculation repairs and portfolio accounting helpers.

Capital convention: each long/short sleeve can use at most one unit per unit
of capital. Industry budgets match across sleeves; unallocated capacity earns
zero excess P&L here. Actual financing, borrow, and trading costs are not modeled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_next_calendar_return(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep gaps missing instead of assigning a later retained month's return."""
    out = frame.copy()
    out["_period"] = pd.to_datetime(out[["year", "month"]].assign(day=1)).dt.to_period("M")
    if out.duplicated(["PERMNO", "_period"]).any():
        raise ValueError("Duplicate PERMNO-month rows: resolve before return alignment.")
    out = out.sort_values(["PERMNO", "_period"], kind="stable")
    next_period = out.groupby("PERMNO")["_period"].shift(-1)
    next_return = out.groupby("PERMNO")["RET"].shift(-1)
    out["return_month"] = out["_period"] + 1
    out["Next_Month_Return"] = next_return.where(next_period.eq(out["return_month"]))
    return out.drop(columns="_period")


def require_known_returns(frame: pd.DataFrame, label: str) -> None:
    """A future missing payoff cannot silently determine portfolio eligibility."""
    values = pd.to_numeric(frame["Next_Month_Return"], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values)
    if invalid.any():
        raise ValueError(
            f"{label}: {int(invalid.sum())} eligible outcomes are unresolved. "
            "Preserve formation eligibility and resolve missing/delisting outcomes; "
            "do not drop these rows or substitute another calendar month."
        )


def drawdown_with_initial_capital(returns: pd.Series) -> pd.Series:
    r = pd.to_numeric(returns, errors="raise")
    if r.isna().any() or not np.isfinite(r).all():
        raise ValueError("Resolve all strategy returns before calculating drawdown.")
    if (r < -1).any():
        raise ValueError("Return below -100%: resolve insolvency/capital accounting.")
    wealth = pd.concat([pd.Series([1.0]), (1 + r).cumprod().reset_index(drop=True)], ignore_index=True)
    return wealth / wealth.cummax() - 1


def annualized_information_ratio(strategy: pd.Series, benchmark: pd.Series) -> float:
    """Annualized active mean / tracking error, aligned by the supplied index."""
    if not strategy.index.equals(benchmark.index):
        raise ValueError("Strategy and benchmark must have identical dated indexes.")
    active = pd.to_numeric(strategy, errors="raise") - pd.to_numeric(benchmark, errors="raise")
    if active.isna().any() or not np.isfinite(active).all():
        raise ValueError("Resolve strategy and benchmark returns before computing IR.")
    tracking_error = active.std(ddof=1)
    if not np.isfinite(tracking_error) or tracking_error == 0:
        return np.nan
    return float(active.mean() / tracking_error * np.sqrt(12))


def validate_evaluation_months(frame: pd.DataFrame) -> None:
    periods = pd.PeriodIndex(pd.to_datetime(frame[["year", "month"]].assign(day=1)), freq="M")
    if periods.duplicated().any() or not periods.is_monotonic_increasing:
        raise ValueError("Performance months must be unique, sorted, and consecutive.")
    if len(periods) > 1 and not np.all(np.diff(periods.asi8) == 1):
        raise ValueError("Performance months must be unique, sorted, and consecutive.")


def winsorize_formation_signal(series: pd.Series) -> pd.Series:
    """Transform the available signal, never the subsequent realized payoff."""
    return series.clip(lower=series.quantile(0.01), upper=series.quantile(0.99))


def allocate_with_cap(raw: pd.Series, budget: float, capacities: pd.Series | float) -> pd.Series:
    """Allocate at most budget, respecting hard capacities and retaining cash.

    No final normalization may undo a cap. Positive raw strength determines
    allocation among names that have not reached capacity.
    """
    raw = pd.to_numeric(raw, errors="raise").astype(float)
    if not raw.index.is_unique or raw.isna().any() or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Allocation strengths must be finite, nonnegative, uniquely indexed.")
    caps = (pd.Series(float(capacities), index=raw.index) if np.isscalar(capacities)
            else capacities.reindex(raw.index).astype(float))
    if caps.isna().any() or not np.isfinite(caps).all() or (caps < 0).any():
        raise ValueError("Capacities must be finite and nonnegative.")
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Budget must be finite and nonnegative.")
    weights = pd.Series(0.0, index=raw.index)
    remaining = min(float(budget), float(caps.sum()))
    free = caps[caps > 0].index
    while remaining > 1e-13 and len(free):
        strengths = raw.loc[free]
        if strengths.sum() == 0:
            strengths = pd.Series(1.0, index=free)
        proposed = remaining * strengths / strengths.sum()
        exceeds = proposed > caps.loc[free]
        if not exceeds.any():
            weights.loc[free] = proposed
            remaining = 0.0
            break
        fixed = free[exceeds.to_numpy()]
        weights.loc[fixed] = caps.loc[fixed]
        remaining -= float(caps.loc[fixed].sum())
        free = free[~exceeds.to_numpy()]
    if (weights > caps + 1e-12).any() or weights.sum() > budget + 1e-10:
        raise AssertionError("Allocation violated a hard constraint.")
    if not np.isclose(weights.sum(), min(budget, caps.sum()), atol=1e-10):
        raise AssertionError("Allocated amount disagrees with feasible budget.")
    return weights


def industry_matched_weights(frame: pd.DataFrame, signal_column: str, cap: float = 0.05):
    """Within-industry quintiles with matched budgets and a hard capital-based cap.

    Industries with fewer than five eligible names do not enter the ranked
    portfolios. Equal industry budgets are limited by the smaller sleeve's
    capacity; unused capital remains unallocated. Ranking never reads returns.
    """
    if not 0 < cap <= 1:
        raise ValueError("Cap must be in (0, 1].")
    if not frame.index.is_unique:
        raise ValueError("Portfolio observations need unique row indexes.")
    chunks = []
    for _, group in frame.groupby("FFInd", sort=True):
        if len(group) < 5:
            continue
        group = group.sort_values("PERMNO", kind="stable").copy()
        ranks = group[signal_column].rank(method="first")
        group["quintile"] = pd.qcut(ranks, 5, labels=["F1", "F2", "F3", "F4", "F5"])
        chunks.append(group)
    if not chunks:
        return frame.iloc[:0].assign(quintile=pd.Series(dtype="object")), pd.Series(dtype=float), pd.Series(dtype=float)
    ranked = pd.concat(chunks)
    longs = ranked[ranked["quintile"] == "F5"]
    shorts = ranked[ranked["quintile"] == "F1"]
    capacity = pd.concat([longs.groupby("FFInd").size(), shorts.groupby("FFInd").size()], axis=1).min(axis=1) * cap
    budgets = allocate_with_cap(pd.Series(1.0, index=capacity.index), 1.0, capacity)
    long_weights = pd.Series(0.0, index=longs.index)
    short_weights = pd.Series(0.0, index=shorts.index)
    for industry, budget in budgets.items():
        lg = longs[longs["FFInd"] == industry][signal_column]
        sg = shorts[shorts["FFInd"] == industry][signal_column]
        long_weights.loc[lg.index] = allocate_with_cap(lg - lg.min() + 1e-12, budget, cap)
        short_weights.loc[sg.index] = allocate_with_cap(sg.max() - sg + 1e-12, budget, cap)
    net = (long_weights.groupby(longs["FFInd"]).sum()
           - short_weights.groupby(shorts["FFInd"]).sum())
    if not np.allclose(net, 0, atol=1e-10):
        raise AssertionError("Industry net exposure is not zero.")
    return ranked, long_weights, short_weights


def enhanced_month(month_data: pd.DataFrame, year: int, month: int, cap: float = 0.05) -> dict:
    """Proposed monthly accounting; missing outcomes stop evaluation explicitly."""
    eligible = month_data.dropna(subset=["Expanded_LM_Signal_Winsorized", "FFInd"]).copy()
    require_known_returns(eligible, f"Enhanced {year}-{month:02d}")
    ranked, lw, sw = industry_matched_weights(eligible, "Expanded_LM_Signal_Winsorized", cap)
    if ranked.empty:
        quintile_returns = pd.Series(np.nan, index=["F1", "F2", "F3", "F4", "F5"])
    else:
        quintile_returns = ranked.groupby("quintile", observed=False)["Next_Month_Return"].mean()
    long_return = float((lw * ranked.loc[lw.index, "Next_Month_Return"]).sum()) if len(lw) else 0.0
    short_underlying_return = float((sw * ranked.loc[sw.index, "Next_Month_Return"]).sum()) if len(sw) else 0.0
    result = {
        "year": year, "month": month,
        "benchmark_return": eligible["Next_Month_Return"].mean(),
        "weighted_F1_return": short_underlying_return, "weighted_F5_return": long_return,
        "long_short_return": long_return - short_underlying_return,
        "num_firms_before_sector_filter": len(eligible), "num_firms_after_sector_filter": len(ranked),
        "num_industries": ranked["FFInd"].nunique(), "F1_count": len(sw), "F5_count": len(lw),
        "long_allocated": float(lw.sum()), "short_allocated": float(sw.sum()),
        "max_name_weight": max(float(lw.max()) if len(lw) else 0.0, float(sw.max()) if len(sw) else 0.0),
        "long_cap_possible": len(lw) * cap >= 1, "short_cap_possible": len(sw) * cap >= 1,
    }
    result.update({f"{q}_return": quintile_returns[q] for q in ["F1", "F2", "F3", "F4", "F5"]})
    result["equal_weight_F5_minus_F1"] = result["F5_return"] - result["F1_return"]
    result["F5_minus_benchmark"] = result["F5_return"] - result["benchmark_return"]
    result["benchmark_minus_F1"] = result["benchmark_return"] - result["F1_return"]
    return result
