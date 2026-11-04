"""Independent timing/identity/missing-outcome checks for phase 2."""
import unittest

import numpy as np
import pandas as pd

from review_data import (COUNT_COLUMNS, attach_identifiers, attach_latest_filing, prepare_return_calendar)
from review_portfolios import evaluate_diagnostics, validation_summary


def filings(rows):
    out = []
    for date, accession, uncertainty in rows:
        row = dict(CIK=1, FILING_DATE=date, ACC_NUM=accession, FORM_TYPE="10-K", N_Words=100, FFInd=1)
        row.update({c: 1 for c in COUNT_COLUMNS})
        row.update(N_Positive=1, N_Negative=3, N_Uncertainty=uncertainty)
        out.append(row)
    return pd.DataFrame(out)


def returns(dates):
    f = pd.DataFrame({"PERMNO": [1] * len(dates), "date": dates,
                      "Ticker": ["OLD"] * len(dates), "COMNAM": ["EXAMPLE INC"] * len(dates),
                      "RET": [0.1] * len(dates)})
    return prepare_return_calendar(f)


def identified(frame):
    return attach_identifiers(frame, mode="legacy_diagnostic",
                              sec_mapping=pd.DataFrame({"ticker": ["OLD"], "cik_str": [1], "title": ["EXAMPLE INC"]}))


class DatedDataRepairs(unittest.TestCase):
    def test_phase2_portfolios_have_known_answer_short_accounting(self):
        f = pd.DataFrame({"PERMNO": range(10), "CIK": [1] * 10, "Ticker": ["OLD"] * 10,
                          "COMNAM": ["EXAMPLE INC"] * 10, "FFInd": [1] * 10,
                          "year": [1994] * 10, "month": [1] * 10,
                          "Baseline_Sentiment": [-5., -4., -3., -2., -1., 1., 2., 3., 4., 5.],
                          "Expanded_LM_Signal": np.arange(10, dtype=float),
                          "Next_Month_Return": [0.2] * 5 + [0.1] * 5,
                          "target_status": ["known"] * 10,
                          "target_return_code": ["known"] * 10, "ACC_NUM": ["first"] * 10})
        monthly, unresolved = evaluate_diagnostics(f, first_year=1994, last_year=1994)
        jan = monthly[monthly.month == 1].set_index("Strategy")
        self.assertAlmostEqual(jan.loc["Baseline", "portfolio_return"], -0.1)
        self.assertAlmostEqual(jan.loc["Expanded", "portfolio_return"], -0.1)
        # Two names per leg at the hard 5% cap: 10% exposure on each side.
        self.assertAlmostEqual(jan.loc["Enhanced", "portfolio_return"], -0.01)
        self.assertAlmostEqual(jan.loc["Enhanced", "long_exposure"], 0.1)
        self.assertTrue(unresolved.empty)

    def test_target_comes_from_full_source_before_ticker_filter(self):
        raw = pd.DataFrame({"PERMNO": [1, 1], "date": ["2020-01-31", "2020-02-28"],
                            "Ticker": ["OLD", "UNMAPPED_NEW"], "COMNAM": ["EXAMPLE INC"] * 2,
                            "RET": [0.1, 0.2]})
        f = identified(prepare_return_calendar(raw))
        self.assertAlmostEqual(f.iloc[0].Next_Month_Return, 0.2)
        self.assertTrue(pd.isna(f.iloc[1].CIK))

    def test_missing_code_is_preserved_without_skipping_month(self):
        raw = pd.DataFrame({"PERMNO": [1] * 3, "date": ["2020-01-31", "2020-02-28", "2020-03-31"],
                            "Ticker": ["OLD"] * 3, "COMNAM": ["EXAMPLE INC"] * 3, "RET": [0.1, "B", 0.3]})
        f = prepare_return_calendar(raw)
        self.assertTrue(pd.isna(f.iloc[0].Next_Month_Return))
        self.assertEqual(f.iloc[0].target_return_code, "B")
        self.assertEqual(f.iloc[0].target_status, "source_return_missing_or_nonnumeric")

    def test_filing_release_lag_blocks_month_end_filing(self):
        f = identified(returns(["2020-01-31", "2020-02-28", "2020-03-31"]))
        p = attach_latest_filing(f, filings([(20200110, "first", 4), (20200228, "second", 4)]))
        self.assertEqual(p.loc[p.month == 2, "ACC_NUM"].iloc[0], "first")
        self.assertEqual(p.loc[p.month == 3, "ACC_NUM"].iloc[0], "second")

    def test_invalid_new_signal_does_not_inherit_old_score(self):
        f = identified(returns(["2020-01-31", "2020-02-28"]))
        p = attach_latest_filing(f, filings([(20200110, "first", 4), (20200214, "new-zero-uncertainty", 0)]))
        feb = p[p.month == 2].iloc[0]
        self.assertEqual(feb.ACC_NUM, "new-zero-uncertainty")
        self.assertTrue(pd.isna(feb.Baseline_Sentiment))
        self.assertAlmostEqual(feb.Guide_Baseline_Sentiment, -2.)

    def test_signal_age_is_calendar_months(self):
        f = identified(returns(["2020-01-31", "2020-04-30", "2020-05-29"]))
        p = attach_latest_filing(f, filings([(20200110, "first", 4)]))
        self.assertTrue(p.loc[p.month == 4, "filing_usable"].iloc[0])
        self.assertFalse(p.loc[p.month == 5, "filing_usable"].iloc[0])
        self.assertTrue(pd.isna(p.loc[p.month == 5, "Baseline_Sentiment"].iloc[0]))

    def test_baseline_parentheses_are_an_explicit_design_choice(self):
        f = identified(returns(["2020-01-31"]))
        p = attach_latest_filing(f, filings([(20200110, "first", 4)]), baseline_formula="published")
        g = attach_latest_filing(f, filings([(20200110, "first", 4)]), baseline_formula="guide")
        self.assertAlmostEqual(p.Baseline_Sentiment.iloc[0], 0.5)
        self.assertAlmostEqual(g.Baseline_Sentiment.iloc[0], -0.4)

    def test_verified_mode_requires_a_real_link_input(self):
        with self.assertRaisesRegex(ValueError, "requires dated"):
            attach_identifiers(returns(["2020-01-31"]), mode="verified_links")

    def test_dated_links_switch_issuer_only_within_effective_dates(self):
        f = returns(["2020-12-31", "2021-01-29"])
        links = pd.DataFrame({"PERMNO": [1, 1], "CIK": [1, 2],
                               "link_start": ["2019-01-01", "2021-01-01"],
                               "link_end": ["2020-12-31", None]})
        out = attach_identifiers(f, mode="verified_links", historical_links=links)
        self.assertEqual(out.CIK.tolist(), [1, 2])
        links.loc[1, "link_start"] = "2020-12-31"
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            attach_identifiers(f, mode="verified_links", historical_links=links)

    def test_missing_outcome_changes_pnl_availability_not_holdings(self):
        f = pd.DataFrame({"PERMNO": range(10), "CIK": [1] * 10, "Ticker": ["OLD"] * 10,
                          "COMNAM": ["EXAMPLE INC"] * 10, "FFInd": [1] * 10,
                          "year": [1994] * 10, "month": [1] * 10,
                          "Baseline_Sentiment": [-5., -4., -3., -2., -1., 1., 2., 3., 4., 5.],
                          "Expanded_LM_Signal": np.arange(10, dtype=float),
                          "Next_Month_Return": [0.1] * 10, "target_status": ["known"] * 10,
                          "target_return_code": ["0.1"] * 10, "ACC_NUM": ["first"] * 10})
        a, _ = evaluate_diagnostics(f, first_year=1994, last_year=1994)
        f.loc[9, "Next_Month_Return"] = np.nan
        f.loc[9, "target_status"] = "absent_source_row"
        f.loc[9, "target_return_code"] = None
        b, unresolved = evaluate_diagnostics(f, first_year=1994, last_year=1994)
        a, b = a[a.month == 1].reset_index(drop=True), b[b.month == 1].reset_index(drop=True)
        for col in ["eligible_security_count", "long_count", "short_count", "long_exposure", "short_exposure", "max_security_weight"]:
            np.testing.assert_allclose(a[col], b[col])
        self.assertTrue(b.portfolio_return.isna().all())
        self.assertEqual(len(unresolved), 3)
        summary = validation_summary(b, identifier_mode="legacy_diagnostic")
        self.assertTrue(summary.full_period_CAGR.isna().all())
        self.assertTrue(summary.performance_reporting_status.str.contains("unverified").all())

    def test_performance_waits_for_complete_outcomes_and_link_input_mode(self):
        m = pd.DataFrame({"Strategy": ["test"] * 12, "year": [2020] * 12, "month": range(1, 13),
                          "portfolio_return": [0.01] * 12, "benchmark_return": [0.005] * 12,
                          "max_security_weight": [0.05] * 12, "max_absolute_industry_net": [0.] * 12,
                          "long_exposure": [1.] * 12, "short_exposure": [1.] * 12})
        blocked = validation_summary(m, identifier_mode="legacy_diagnostic").iloc[0]
        self.assertTrue(pd.isna(blocked.full_period_CAGR))
        ready = validation_summary(m, identifier_mode="verified_links").iloc[0]
        self.assertAlmostEqual(ready.full_period_CAGR, 1.01 ** 12 - 1)
        m.loc[3, "portfolio_return"] = np.nan
        blocked = validation_summary(m, identifier_mode="verified_links").iloc[0]
        self.assertTrue(pd.isna(blocked.full_period_CAGR))


if __name__ == "__main__":
    unittest.main(verbosity=2)
