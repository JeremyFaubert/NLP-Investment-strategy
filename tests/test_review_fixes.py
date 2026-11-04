"""Known-answer and controlled integration tests for the proposed repairs.

Run: python3 -m unittest -v test_review_fixes.py
These tests do not replace a backtest on the original user data.
"""
import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from review_fixes import (
    add_next_calendar_return, allocate_with_cap, annualized_information_ratio,
    drawdown_with_initial_capital, enhanced_month, industry_matched_weights,
    require_known_returns, validate_evaluation_months, winsorize_formation_signal,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = json.loads((ROOT / "tests/fixtures/phase1_calculation_candidate.ipynb").read_text())
ORIGINAL = json.loads((ROOT / "archive/NLP_Sentiment_Investing_model.ipynb").read_text())


def cell(nb, n):
    return "".join(nb["cells"][n - 1]["source"])


def env_base():
    return dict(pd=pd, np=np, require_known_returns=require_known_returns,
                annualized_information_ratio=annualized_information_ratio,
                drawdown_with_initial_capital=drawdown_with_initial_capital,
                validate_evaluation_months=validate_evaluation_months,
                winsorize_formation_signal=winsorize_formation_signal,
                enhanced_month=enhanced_month)


def execute(source, env):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(source, env)


def two_industries(n=100):
    return pd.DataFrame({
        "PERMNO": np.arange(2 * n), "Ticker": [f"T{i}" for i in range(2 * n)],
        "FFInd": [1] * n + [2] * n,
        "Expanded_LM_Signal_Winsorized": np.r_[np.arange(n), 1000 + np.arange(n)].astype(float),
        "Next_Month_Return": [0.1] * n + [-0.2] * n,
    })


class CalculationRepairs(unittest.TestCase):
    def test_current_bug_and_candidate_short_pnl(self):
        f = pd.DataFrame({"year": [1994, 1994], "month": [1, 1], "PERMNO": [1, 2],
                          "Ticker": ["L", "S"], "Baseline_Sentiment": [1., -1.],
                          "Next_Month_Return": [0.1, 0.2]})
        old, new = env_base(), env_base()
        old["df_backtest"], new["df_backtest"] = f.copy(), f.copy()
        execute(cell(ORIGINAL, 17), old)
        execute(cell(CANDIDATE, 17), new)
        self.assertAlmostEqual(old["baseline_results"].iloc[0]["long_short_return"], 0.30)
        self.assertAlmostEqual(new["baseline_results"].iloc[0]["long_short_return"], -0.10)

    def test_gaps_are_not_bridged(self):
        f = pd.DataFrame({"PERMNO": [1, 1, 2, 2], "year": [2020] * 4,
                          "month": [1, 3, 1, 2], "RET": [0.01, 0.03, 0.02, 0.04]})
        out = add_next_calendar_return(f).set_index(["PERMNO", "month"])
        self.assertTrue(pd.isna(out.loc[(1, 1), "Next_Month_Return"]))
        self.assertAlmostEqual(out.loc[(2, 1), "Next_Month_Return"], 0.04)
        self.assertEqual(out.loc[(1, 1), "return_month"], pd.Period("2020-02"))

    def test_duplicate_security_month_rejected(self):
        f = pd.DataFrame({"PERMNO": [1, 1], "year": [2020, 2020], "month": [1, 1], "RET": [0., 0.]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            add_next_calendar_return(f)

    def test_missing_future_outcome_does_not_select_universe(self):
        f = pd.DataFrame({"year": [1994, 1994], "month": [1, 1], "PERMNO": [1, 2],
                          "Ticker": ["L", "S"], "Baseline_Sentiment": [1., -1.],
                          "Next_Month_Return": [0.1, np.nan]})
        env = env_base()
        env["df_backtest"] = f
        with self.assertRaisesRegex(ValueError, "unresolved"):
            execute(cell(CANDIDATE, 17), env)

    def test_initial_loss_is_drawdown(self):
        self.assertAlmostEqual(drawdown_with_initial_capital(pd.Series([-0.1, 0.05])).min(), -0.1)
        self.assertAlmostEqual(drawdown_with_initial_capital(pd.Series([0.1, -0.2, 0.05])).min(), -0.2)
        self.assertAlmostEqual(drawdown_with_initial_capital(pd.Series([-1., 0.])).min(), -1.)

    def test_insolvency_is_not_compounded_silently(self):
        with self.assertRaisesRegex(ValueError, "insolvency"):
            drawdown_with_initial_capital(pd.Series([-1.2, 0.3]))

    def test_information_ratio_uses_benchmark_and_annualizes(self):
        s = pd.Series([0.1, 0.2, 0.3])
        first = annualized_information_ratio(s, pd.Series([0.1] * 3))
        second = annualized_information_ratio(s, pd.Series([0.] * 3))
        self.assertAlmostEqual(first, np.sqrt(12))
        self.assertAlmostEqual(second, 2 * np.sqrt(12))
        with self.assertRaisesRegex(ValueError, "identical"):
            annualized_information_ratio(s, pd.Series([0., 0., 0.], index=[1, 2, 3]))

    def test_cap_keeps_cash_when_infeasible(self):
        w = allocate_with_cap(pd.Series([1., 3.]), 1., 0.05)
        np.testing.assert_allclose(w, [0.05, 0.05])
        self.assertAlmostEqual(1 - w.sum(), 0.90)

    def test_feasible_cap_allocation_known_answer(self):
        w = allocate_with_cap(pd.Series([1., 3., 100.]), 1., 0.5)
        np.testing.assert_allclose(w, [0.125, 0.375, 0.5])

    def test_industry_exposure_matches_with_position_caps(self):
        f = two_industries()
        ranked, lw, sw = industry_matched_weights(f, "Expanded_LM_Signal_Winsorized")
        longs = lw.groupby(ranked.loc[lw.index, "FFInd"]).sum()
        shorts = sw.groupby(ranked.loc[sw.index, "FFInd"]).sum()
        np.testing.assert_allclose(longs, [0.5, 0.5])
        np.testing.assert_allclose(shorts, [0.5, 0.5])
        self.assertLessEqual(max(lw.max(), sw.max()), 0.05 + 1e-12)
        row = enhanced_month(f, 1994, 1)
        self.assertAlmostEqual(row["long_short_return"], 0.)

    def test_small_industries_partial_allocation_known_pnl(self):
        f = two_industries(5)
        f["Next_Month_Return"] = [0.1, 0., 0., 0., 0.2, 0.3, 0., 0., 0., 0.4]
        row = enhanced_month(f, 1994, 1)
        self.assertAlmostEqual(row["long_allocated"], 0.1)
        self.assertAlmostEqual(row["short_allocated"], 0.1)
        self.assertAlmostEqual(row["weighted_F5_return"], 0.03)
        self.assertAlmostEqual(row["weighted_F1_return"], 0.02)
        self.assertAlmostEqual(row["long_short_return"], 0.01)
        self.assertLessEqual(row["max_name_weight"], 0.05)

    def test_ineligible_industries_do_not_create_uncapped_holdings(self):
        row = enhanced_month(two_industries(4), 1994, 1)
        self.assertEqual(row["long_allocated"], 0.)
        self.assertEqual(row["short_allocated"], 0.)
        self.assertEqual(row["long_short_return"], 0.)

    def test_realized_tail_payoffs_remain_unchanged(self):
        f = pd.DataFrame({"year": [1994] * 200, "month": [1] * 200,
                          "Expanded_LM_Signal": list(range(199)) + [10000.],
                          "Next_Month_Return": [-0.5] + [0.01] * 198 + [1.]})
        env = env_base()
        env["df_backtest"] = f.copy()
        execute(cell(CANDIDATE, 19), env)
        out = env["df_enhanced"]
        pd.testing.assert_series_equal(out["Next_Month_Return"], f["Next_Month_Return"])
        self.assertLess(out["Expanded_LM_Signal_Winsorized"].max(), 10000.)

    def test_candidate_metrics_include_initial_capital(self):
        env = env_base()
        execute(cell(CANDIDATE, 21), env)
        results = pd.DataFrame({"year": [1994] * 3, "month": [1, 2, 3],
                                "long_short_return": [-0.1, 0.05, 0.02], "benchmark_return": [0.] * 3})
        signals = pd.DataFrame({"year": [1994] * 6, "month": [1, 1, 2, 2, 3, 3],
                                "signal": [1., 2.] * 3, "ret": [0.01, 0.02] * 3})
        m = env["calculate_strategy_metrics"]("test", results, signals, "signal", "ret")
        m = dict(zip(m["Metric"], m["Value"]))
        self.assertAlmostEqual(m["Maximum Drawdown"], -0.1)
        self.assertAlmostEqual(m["Cumulative Return"], 0.9 * 1.05 * 1.02 - 1)

    def test_missing_duplicate_or_unsorted_evaluation_months_rejected(self):
        for months in [[1, 3], [1, 1], [2, 1]]:
            with self.assertRaisesRegex(ValueError, "consecutive"):
                validate_evaluation_months(pd.DataFrame({"year": [1994, 1994], "month": months}))
        validate_evaluation_months(pd.DataFrame({"year": [1994, 1995], "month": [12, 1]}))

    def test_plot_includes_starting_capital_and_holding_month(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f = pd.DataFrame({"year": [1994] * 3, "month": [1, 2, 3], "long_short_return": [0.1, -0.1, 0.05]})
        env = env_base()
        env.update(baseline_results=f.copy(), main_results=f.copy(), enhanced_results=f.copy())
        execute(cell(CANDIDATE, 24), env)
        frame = env["baseline_plot"]
        self.assertEqual(frame["date"].iloc[0], pd.Timestamp("1994-01-31"))
        self.assertEqual(frame["date"].iloc[1], pd.Timestamp("1994-02-28"))
        self.assertEqual(frame["wealth_index"].iloc[0], 1.0)
        self.assertAlmostEqual(frame["wealth_index"].iloc[-1], 1.1 * 0.9 * 1.05)
        plt.close("all")

    def test_equal_weight_and_weighted_table_spreads_are_distinct(self):
        f = pd.DataFrame({"F1_return": [0.011561], "F2_return": [0.], "F3_return": [0.],
                          "F4_return": [0.], "F5_return": [0.008975],
                          "weighted_F1_return": [0.012103], "weighted_F5_return": [0.008366],
                          "long_short_return": [0.008366 - 0.012103]})
        env = env_base()
        env.update(main_results=f.copy(), enhanced_results=f.copy())
        execute(cell(CANDIDATE, 29), env)
        row = env["fractile_table"].iloc[1]
        self.assertAlmostEqual(row["Equal-weight F5 - F1"], -0.002586)
        self.assertAlmostEqual(row["Portfolio spread"], -0.003737)

    def test_72_month_no_alpha_control_panel(self):
        frames = []
        for t in range(72):
            f = two_industries()
            f["year"], f["month"] = 1994 + t // 12, t % 12 + 1
            f["Baseline_Sentiment"] = ((np.arange(200) % 20) - 9.5) / 10
            f["Expanded_LM_Signal"] = f["Expanded_LM_Signal_Winsorized"]
            f["Next_Month_Return"] = 0.01 + 0.005 * np.sin(t)
            frames.append(f)
        fixture = pd.concat(frames, ignore_index=True)
        old, new = env_base(), env_base()
        old["df_backtest"], new["df_backtest"] = fixture.copy(), fixture.copy()
        execute(cell(ORIGINAL, 17), old)
        for n in [17, 18, 19, 20]:
            execute(cell(CANDIDATE, n), new)
        original = old["baseline_results"]["long_short_return"]
        for label in ["baseline_results", "main_results", "enhanced_results"]:
            self.assertEqual(len(new[label]), 72)
            self.assertLess(new[label]["long_short_return"].abs().max(), 1e-12)
        expected_old = 2 * np.mean([0.01 + 0.005 * np.sin(t) for t in range(72)])
        self.assertAlmostEqual(original.mean(), expected_old)
        self.assertLessEqual(new["enhanced_results"]["max_name_weight"].max(), 0.05 + 1e-10)

    def test_every_candidate_cell_compiles_and_old_outputs_removed(self):
        for i, c in enumerate(CANDIDATE["cells"], 1):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"candidate cell {i}", "exec")
                self.assertEqual(c["outputs"], [])
                self.assertIsNone(c["execution_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
