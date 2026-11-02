"""Descriptive month-fixed-effect regressions with issuer/month clustering.

Known outcomes are used only for a labeled conditional-association analysis.
This is not a substitute for resolving missing portfolio payoffs or validating
historical issuer links, and it is not an out-of-sample test.
"""
import gc
import numpy as np
import pandas as pd


def descriptive_regressions(panel, *, first_year=1994, last_year=2023):
    import statsmodels.formula.api as smf

    data = panel[panel.year.between(first_year, last_year)].copy()
    data = data[data.Baseline_Sentiment.notna() & data.Expanded_LM_Signal.notna() & data.FFInd.notna()]
    data["year_month"] = data.year.astype(str) + "-" + data.month.astype(str).str.zfill(2)
    rows = []
    for signal in ["Baseline_Sentiment", "Expanded_LM_Signal"]:
        observed = data[["Next_Month_Return", signal, "year_month", "CIK"]].dropna()
        fit = smf.ols(f"Next_Month_Return ~ {signal} + C(year_month)", data=observed).fit()
        groups = np.column_stack([pd.factorize(observed.CIK)[0], pd.factorize(observed.year_month)[0]])
        robust = fit.get_robustcov_results(cov_type="cluster", groups=groups,
                                           use_correction=True, df_correction=True, use_t=True)
        i = fit.params.index.get_loc(signal)
        y = observed.Next_Month_Return
        within_sst = ((y - observed.groupby("year_month").Next_Month_Return.transform("mean")) ** 2).sum()
        total_sst = ((y - y.mean()) ** 2).sum()
        month_only_r2 = 1 - within_sst / total_sst
        rows.append({"signal": signal, "observations": len(observed),
                     "issuers": observed.CIK.nunique(), "months": observed.year_month.nunique(),
                     "coefficient": float(fit.params[signal]),
                     "conventional_p": float(fit.pvalues[signal]),
                     "issuer_month_clustered_t": float(robust.tvalues[i]),
                     "issuer_month_clustered_p": float(robust.pvalues[i]),
                     "clustered_95pct_ci_low": float(robust.conf_int()[i, 0]),
                     "clustered_95pct_ci_high": float(robust.conf_int()[i, 1]),
                     "full_model_r2": float(fit.rsquared), "month_only_r2": float(month_only_r2),
                     "incremental_r2_from_signal": float(fit.rsquared - month_only_r2),
                     "interpretation": "Conditional association in the supplied linked sample; observed outcomes only; no out-of-sample or causal claim."})
        del fit, robust, groups
        gc.collect()
    return pd.DataFrame(rows)
