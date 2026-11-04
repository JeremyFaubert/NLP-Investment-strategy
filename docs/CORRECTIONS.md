# Calculation and interpretation corrections

The original implementation does not establish a profitable sentiment-based long-short strategy. The original report, saved notebook and figures are retained under `archive/` as superseded evidence. The root notebook is the current research entry point.

## Portfolio accounting and outcome treatment

The original baseline assigns negative short weights, then subtracts their already signed P&L. That expression adds the returns of two long groups. With one capital unit assigned to each group, it is equivalent to 200% long exposure before financing. The corrected portfolio module uses positive short magnitudes and subtracts the underlying shorted-stock payoff exactly once. The archived calculation fixture instead keeps negative weights and adds signed P&L; both conventions give the same intended spread.

The original enhanced model clips future realized returns while the write-up describes clipping sentiment. Changing an eventual payoff cannot be interpreted as an executable portfolio risk control. The current enhanced model winsorizes the formation-time signal and preserves actual realized returns.

Ranking within industries does not by itself make industry exposures neutral. The revised enhanced allocation matches industry budgets across its long and short sleeves. Each position has a hard 5% cap relative to modeled capital. Industries with fewer than five eligible names do not participate. When caps prevent full investment, remaining capacity stays unallocated and contributes zero excess P&L. A final renormalization never bypasses the cap. These are explicit revised design choices.

Each sleeve can use at most one unit per unit of capital, allowing up to 200% gross exposure. Baseline and expanded comparators are intentionally uncapped. When either baseline sleeve is absent, both sleeves remain unallocated. Borrow availability, fees, financing, collateral interest and trading costs are not modeled, so the outputs are not implementable after-cost returns.

## Data identity, chronology and missing outcomes

The original current-ticker-to-CIK join is not a historical security-to-issuer link. Ticker reuse can assign historical stock records to a different issuer. The current snapshot has no effective-dated validity intervals. A count of PERMNOs mapped to multiple CIKs would be a diagnostic, not an estimated error rate. The CIK-CUSIP input cannot repair the supplied return schema because it contains no CUSIP.

Default `legacy_diagnostic` mode retains the frozen current-ticker cohort solely for inspection and blocks full-period performance reporting. Alternative `verified_links` mode requires a dated PERMNO-CIK table and rejects overlapping intervals. That mode name means a link input has been supplied; it does not independently certify provenance or coverage. Review the actual historical relationships before using it.

The original `shift(-1)` is applied after filtering return rows and can jump across calendar gaps. The current pipeline pairs targets with exactly the next calendar month from the full source table before identifier or missing-return filtering. Absent or nonnumeric source payoffs remain unresolved.

The original independent column forward fills can exceed the intended calendar age and mix information from different filings. The replacement selects a single whole latest filing record, retains its accession number, and requires an age of at most three calendar months. A disclosed one Monday-Friday business-day lag approximates availability when acceptance timestamps are absent. It is not a holiday-aware exchange calendar or proof of an executable closing-price trade.

Holdings are formed before future payoff availability is checked. An unresolved held payoff makes the portfolio month's return unavailable; it is not replaced with zero, skipped to a later month, or removed to change the portfolio. The common formation universe requires a known current-month return, usable baseline and expanded scores, and an FFInd value. Enhanced ranking imposes its additional industry-size rule.

Confirm source extract coverage against the original query and expected source counts. A long date window or raw observation count alone does not prove a complete export. Obtain missing/terminal payoff documentation and applicable delisting records before full validation.

## Metrics, sample and inference

Drawdown includes starting capital. The information-ratio helper uses aligned strategy-minus-benchmark returns and tracking error, annualized at monthly frequency. The Sharpe statistic is labeled with its zero cash baseline. Missing months, duplicate months and insolvency are not silently compounded into an apparently valid complete path.

The current analysis retains all 360 formation months, January 1994 through December 2023. Their nominal holding months are February 1994 through January 2024. The original enhanced implementation reports only 359 months. Full-period strategy metrics are withheld while issuer identity, held payoffs or benchmark outcomes remain unresolved.

Descriptive regressions include month effects and report two-way issuer/month clustered uncertainty. They use observed outcomes only, so their complete-case associations do not repair missing held payoffs. Full-model R-squared includes the month effects; incremental R-squared measures the addition of the signal to a month-only model. Neither a small p-value nor an association in the linked cohort establishes causality, profitability or market-wide prediction.

The original enhanced quintile table mixes equal-weight quintile returns with a signal-weighted portfolio spread. The calculation helpers now separate those quantities. The misleading original table is not reused as a current result.

## Research claims and reproducibility

No precommitted holdout or walk-forward evaluation has been demonstrated, so an out-of-sample claim is unsupported. The experiment changes several accounting, sample and allocation treatments together and cannot establish that complexity caused worse performance. Underpricing, contrarian trading and alternative sentiment models remain hypotheses, not demonstrated explanations or validated profitable alternatives.

The project uses precomputed Loughran-McDonald filing summaries. It does not perform raw filing extraction or dictionary matching within this notebook. The published baseline formula `(P-N)/U + 1` is retained for controlled comparison. The separate `guide` option `(P-N)/(U+1)` is a different signal specification and may change portfolio participation; it must not be substituted silently.

The public repository contains a frozen public SEC response, corrected code and synthetic calculation tests. Private source CSVs, fingerprints, metadata and financial output records are not published. Financial-history reproduction requires access to appropriate source inputs; public tests verify arithmetic and safeguards without them. See `RESULTS.md` for output interpretation and remaining requirements.

Primary references: [LM summary source](https://sraf.nd.edu/sec-edgar-data/), [SEC current ticker data](https://www.sec.gov/files/company_tickers.json), [Microsoft worksheet limits](https://support.microsoft.com/en-us/excel/excel-specifications-and-limits), [statsmodels robust covariance](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html).
