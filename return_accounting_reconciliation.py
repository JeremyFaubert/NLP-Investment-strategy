"""Reconcile the original positive baseline with the intended long-short P&L.

This is an accounting diagnostic on original monthly outputs. It deliberately
does not repair historical identities, universe selection or target alignment.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def reconcile(original: pd.DataFrame):
    f = original.copy()
    required = ['year', 'month', 'long_return', 'short_return', 'long_short_return', 'benchmark_return']
    if not set(required).issubset(f.columns):
        raise ValueError('Original baseline output columns are missing.')
    p = pd.PeriodIndex(pd.to_datetime(f[['year', 'month']].assign(day=1)), freq='M')
    if p.has_duplicates or not p.is_monotonic_increasing or not np.all(np.diff(p.asi8) == 1):
        raise ValueError('Monthly output must be unique, consecutive and sorted.')
    if f[required].isna().any().any() or not np.isfinite(f[required]).all().all():
        raise ValueError('Resolve missing or nonfinite original monthly outputs first.')
    np.testing.assert_allclose(f.long_short_return, f.long_return - f.short_return, rtol=1e-12, atol=1e-12)
    f['formation_month'] = p.astype(str)
    f['original_target_month_label'] = (p + 1).astype(str)
    f['positive_score_group_held_long'] = f.long_return
    f['negative_score_group_held_long'] = -f.short_return
    f['original_200pct_long_diagnostic'] = f.long_return - f.short_return
    f['same_basket_100pct_long_diagnostic'] = (f.long_return - f.short_return) / 2
    f['corrected_long_short_sign_only'] = f.long_return + f.short_return
    np.testing.assert_allclose(f.original_200pct_long_diagnostic,
                               2 * f.same_basket_100pct_long_diagnostic, rtol=1e-12, atol=1e-12)
    # This identity is monthly. Annualized compounded returns must not simply
    # be halved or doubled, because compounding is nonlinear.
    rows = []
    specifications = [
        ('Original reported baseline', 'original_200pct_long_diagnostic', 2., 0., 2.),
        ('Same two groups, 100% long', 'same_basket_100pct_long_diagnostic', 1., 0., 1.),
        ('Notebook equal-weight comparison basket', 'benchmark_return', 1., 0., 1.),
        ('Correct short sign only', 'corrected_long_short_sign_only', 1., 1., 0.),
        ('Positive-score group held long', 'positive_score_group_held_long', 1., 0., 1.),
        ('Negative-score group held long', 'negative_score_group_held_long', 1., 0., 1.),
    ]
    for label, column, long_exposure, short_exposure, net_exposure in specifications:
        returns = f[column]
        if returns.le(-1).any():
            raise ValueError('Liquidation/insolvency needs explicit capital accounting.')
        wealth = float((1 + returns).prod())
        path = np.r_[1., (1 + returns).cumprod().to_numpy()]
        drawdown = path / np.maximum.accumulate(path) - 1
        rows.append({'interpretation': label, 'monthly_column': column,
                     'months': len(f), 'long_exposure_per_capital': long_exposure,
                     'short_exposure_per_capital': short_exposure, 'net_exposure_per_capital': net_exposure,
                     'average_monthly_return': float(returns.mean()),
                     'annualized_compounded_return': wealth ** (12 / len(f)) - 1,
                     'cumulative_return': wealth - 1, 'ending_wealth_per_unit': wealth,
                     'hit_rate': float(returns.gt(0).mean()), 'maximum_drawdown_including_initial_capital': float(drawdown.min()),
                     'status': 'Legacy-sample accounting diagnostic only; identities, selection and target alignment unverified; costs excluded'})
    return f, pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_csv', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    monthly, summary = reconcile(pd.read_csv(args.baseline_csv))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.output_dir / 'monthly_accounting_reconciliation.csv', index=False)
    summary.to_csv(args.output_dir / 'return_interpretation_summary.csv', index=False)
    print(summary[['interpretation', 'average_monthly_return', 'annualized_compounded_return', 'hit_rate']].to_string(index=False))
