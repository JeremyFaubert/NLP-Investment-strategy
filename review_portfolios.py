"""Formation-first portfolio diagnostics; incomplete payoffs block performance."""
import numpy as np
import pandas as pd

from review_fixes import (annualized_information_ratio, drawdown_with_initial_capital,
                          industry_matched_weights, validate_evaluation_months, winsorize_formation_signal)


def _weights(month, strategy):
    empty = pd.Series(dtype=float)
    if strategy == "Baseline":
        long = month.loc[month.Baseline_Sentiment > 0, "Baseline_Sentiment"]
        short = -month.loc[month.Baseline_Sentiment < 0, "Baseline_Sentiment"]
        if long.empty or short.empty:
            return empty, empty
        return long / long.sum(), short / short.sum()
    if strategy == "Expanded":
        if len(month) < 5:
            return empty, empty
        ranks = month.Expanded_LM_Signal.rank(method="first")
        quintiles = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5])
        li, si = month.index[quintiles == 5], month.index[quintiles == 1]
        return pd.Series(1 / len(li), index=li), pd.Series(1 / len(si), index=si)
    if strategy == "Enhanced":
        _, long, short = industry_matched_weights(month, "Expanded_LM_Signal_Winsorized", cap=0.05)
        return long, short
    raise ValueError("Unknown portfolio strategy.")


def evaluate_diagnostics(panel, *, first_year=1994, last_year=2023):
    """Keep unresolved held returns as NaN; never change eligibility using them.

    All three specifications start from a common formation-time universe.
    When either baseline sleeve is absent, both sleeves remain unallocated.
    Enhanced allocations retain capacity that cannot meet the cap in cash.
    """
    source = panel[panel.year.between(first_year, last_year)].copy()
    source = source[source.Baseline_Sentiment.notna() & source.Expanded_LM_Signal.notna() & source.FFInd.notna()]
    source["Expanded_LM_Signal_Winsorized"] = source.groupby(["year", "month"])["Expanded_LM_Signal"].transform(winsorize_formation_signal)
    records, unresolved = [], []
    for period in pd.period_range(f"{first_year}-01", f"{last_year}-12", freq="M"):
        month = source[(source.year == period.year) & (source.month == period.month)].sort_values("PERMNO", kind="stable")
        for strategy in ["Baseline", "Expanded", "Enhanced"]:
            long, short = _weights(month, strategy)
            lr = month.loc[long.index, "Next_Month_Return"]
            sr = month.loc[short.index, "Next_Month_Return"]
            unknown_l = float(long[lr.isna()].sum())
            unknown_s = float(short[sr.isna()].sum())
            known_l = float((long[lr.notna()] * lr[lr.notna()]).sum())
            known_s = float((short[sr.notna()] * sr[sr.notna()]).sum())
            if len(month):
                lg = long.groupby(month.loc[long.index, "FFInd"]).sum()
                sg = short.groupby(month.loc[short.index, "FFInd"]).sum()
                sector_net = lg.subtract(sg, fill_value=0)
                sector_max = float(sector_net.abs().max()) if len(sector_net) else 0.
            else:
                sector_max = 0.
            portfolio_known = unknown_l == 0 and unknown_s == 0
            benchmark_known = bool(len(month)) and month.Next_Month_Return.notna().all()
            records.append({
                "Strategy": strategy, "year": period.year, "month": period.month,
                "formation_month": str(period), "return_month": str(period + 1),
                "eligible_security_count": len(month), "long_count": int((long > 0).sum()),
                "short_count": int((short > 0).sum()), "long_exposure": float(long.sum()),
                "short_exposure": float(short.sum()),
                "max_security_weight": max(float(long.max()) if len(long) else 0., float(short.max()) if len(short) else 0.),
                "max_absolute_industry_net": sector_max,
                "unknown_long_weight": unknown_l, "unknown_short_weight": unknown_s,
                "portfolio_return": known_l - known_s if portfolio_known else np.nan,
                "benchmark_return": month.Next_Month_Return.mean() if benchmark_known else np.nan,
                "return_status": "complete" if portfolio_known else "unresolved_held_outcome",
                "no_positions": not len(long) and not len(short),
            })
            for side, weights, returns in [("long", long, lr), ("short", short, sr)]:
                indexes = returns.index[returns.isna() & weights.gt(0)]
                for idx in indexes:
                    row = month.loc[idx]
                    unresolved.append({"Strategy": strategy, "formation_month": str(period),
                        "return_month": str(period + 1), "side": side, "weight": float(weights.loc[idx]),
                        "PERMNO": row.PERMNO, "CIK": row.CIK, "Ticker": row.Ticker,
                        "COMNAM": row.COMNAM, "target_status": row.target_status,
                        "target_return_code": row.target_return_code, "ACC_NUM": row.ACC_NUM})
    monthly = pd.DataFrame(records)
    exceptions = pd.DataFrame(unresolved)
    return monthly, exceptions


def validation_summary(monthly, *, identifier_mode):
    rows = []
    for strategy, group in monthly.groupby("Strategy", sort=False):
        incomplete = int(group.portfolio_return.isna().sum())
        reasons = []
        if identifier_mode != "verified_links":
            reasons.append("historical identifiers unverified")
        if incomplete:
            reasons.append(f"{incomplete} months have unresolved held outcomes")
        if group.benchmark_return.isna().any():
            reasons.append("benchmark outcomes incomplete")
        if group.portfolio_return.le(-1).any():
            reasons.append("insolvency requires explicit liquidation accounting")
        row = {"Strategy": strategy, "months": len(group),
            "complete_portfolio_months": int(group.portfolio_return.notna().sum()),
            "incomplete_portfolio_months": incomplete,
            "max_security_weight": float(group.max_security_weight.max()),
            "max_absolute_industry_net": float(group.max_absolute_industry_net.max()),
            "minimum_long_exposure": float(group.long_exposure.min()),
            "minimum_short_exposure": float(group.short_exposure.min()),
            "performance_reporting_status": "; ".join(reasons) if reasons else "data checks pass under supplied link/capital assumptions",
            "full_period_CAGR": np.nan,
            "average_monthly_return": np.nan, "cumulative_return": np.nan,
            "maximum_drawdown": np.nan, "annualized_sharpe_zero_cash_baseline": np.nan,
            "annualized_information_ratio": np.nan, "hit_rate": np.nan,
        }
        if not reasons:
            validate_evaluation_months(group)
            ret = group.portfolio_return
            wealth = (1 + ret).prod()
            vol = ret.std(ddof=1)
            row.update(full_period_CAGR=wealth ** (12 / len(ret)) - 1,
                       average_monthly_return=ret.mean(), cumulative_return=wealth - 1,
                       maximum_drawdown=drawdown_with_initial_capital(ret).min(),
                       annualized_sharpe_zero_cash_baseline=ret.mean() / vol * np.sqrt(12) if vol > 0 else np.nan,
                       annualized_information_ratio=annualized_information_ratio(ret, group.benchmark_return),
                       hit_rate=ret.gt(0).mean())
        rows.append(row)
    # Numerical summaries under supplied data assumptions do not establish
    # out-of-sample performance, financing feasibility, or a market-wide edge.
    return pd.DataFrame(rows)
