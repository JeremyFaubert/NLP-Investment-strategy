# SEC Filing Sentiment and Equity Returns

**Authors: Jeremy Faubert and Matthias Baker**

**Course: FT370 - Investment Applications of NLP**

A Python research project comparing three sentiment-based portfolio specifications using precomputed Loughran-McDonald SEC 10-K/10-Q summaries and monthly equity returns.

**Research status:** the original profitability and out-of-sample conclusions are withdrawn. The baseline contains a short-position sign error, and the enhanced implementation changes future payoffs. The revised pipeline corrects the accounting and adds explicit data-quality safeguards. Complete strategy performance requires validated historical issuer links, complete return coverage and resolution of missing held payoffs.

## Confirmed calculation findings

- The original baseline subtracts an already signed short P&L. Its expression adds the returns of two long groups, equivalent to 200% long exposure under its stated capital convention before financing.
- In a synthetic example, a long stock rising 10% and a shorted stock rising 20% should produce -10%; the original reports +30%. If the shorted stock falls 20%, the correction raises the result from -10% to +30%. Correct accounting does not force a negative outcome.
- The original enhanced implementation clips future realized returns. The corrected version transforms formation-time signals and preserves actual payoffs.
- The current public evidence does not establish investment viability, out-of-sample prediction, underpricing, or that added complexity caused worse performance.

See [calculation interpretation](docs/RESULTS.md) and [the correction notice](docs/CORRECTIONS.md). The original public report, figures and saved notebook are retained as [superseded evidence](archive/README.md).

## Current implementation

The [root notebook](NLP_Sentiment_Investing_model.ipynb) is the current entry point. Its outputs are cleared so stale historical results cannot appear beside corrected code.

1. Pair exactly the next calendar month's return from the full source table before filtering; preserve missing codes and absent outcomes.
2. Select one whole latest filing record, retain accession provenance and enforce a three-calendar-month age limit. A disclosed one Monday-Friday business-day availability lag approximates timing when acceptance timestamps are absent.
3. Form holdings using formation-time information. Baseline uses signed-score groups; expanded uses global quintiles; enhanced uses within-industry quintiles, matched industry allocations and a hard 5% capital-based name cap.
4. Winsorize the formation signal rather than realized payoffs. Keep infeasible capacity unallocated and report unresolved held payoffs explicitly.
5. Suppress complete performance statistics while identity, payoff or benchmark checks fail. Descriptive regressions report issuer/month clustered uncertainty and incremental R-squared beyond month effects.

Default legacy_diagnostic mode uses a frozen public current-ticker snapshot for inspection and cannot certify historical identities. Alternative verified_links mode requires an independently reviewed dated PERMNO-CIK table; supplying it is not independent certification. The CIK-CUSIP input cannot repair returns that lack historical CUSIP.

Each side can use at most one unit per capital unit (up to 200% gross); unallocated capacity contributes zero excess P&L. Trading costs, borrow availability, fees and financing are excluded. Industry-size, cash and formula choices are documented research conventions.

## Run the public calculation checks

Python 3.12 was used for the recorded checks. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Windows, activate with .venv\Scripts\activate. All 30 focused checks use public synthetic fixtures and need no private financial inputs. They cover short accounting, calendar gaps, filing chronology, future-outcome invariance, caps, industry matching and metric definitions. Passing them does not certify a historical dataset. Some checks execute labeled intermediate calculation fixtures; the root notebook contains the current data pipeline.

## Run the research notebook with source data

Raw source files are not redistributed. Obtain appropriate inputs under the necessary usage rights and follow [data/README.md](data/README.md). Keep source provenance and any fingerprints private. This public revision does not publish metadata, fingerprints or financial output records from private Drive inputs. The [frozen SEC snapshot](resources/README.md) is a public source, not a private Drive input or a historical issuer link.

Start Jupyter in the repository root and open NLP_Sentiment_Investing_model.ipynb. Set NLP_DATA_DIR for a private data location and NLP_OUTPUT_DIR for a separate output directory. Defaults are data/ and research_outputs/. In Colab, clone the repository, change directory to it and configure the data path; enable the optional Drive mount only if needed. Local code-cell execution was checked; Colab mounting/frontend integration was not tested.

Dense month-effect regressions can require several GB RAM. Set RUN_PANEL_REGRESSIONS=False to inspect data and portfolio diagnostics without those fits.

## Repository layout

- NLP_Sentiment_Investing_model.ipynb: current guarded research pipeline.
- review_data.py, review_portfolios.py, review_fixes.py, review_inference.py: data, allocation, accounting and descriptive-inference helpers.
- tests/: known-answer checks and labeled intermediate calculation fixtures.
- resources/: the frozen public SEC response and source description.
- results/: research-status notice; no private financial outputs.
- docs/: correction notice and calculation interpretation.
- archive/: superseded original public notebook, report and figures.

The project starts from upstream LM summary counts; raw text extraction and dictionary matching are not performed here. Baseline retains (Positive-Negative)/Uncertainty + 1 for comparison. The alternative stabilized denominator is a distinct specification. Clean-data testing and a precommitted evaluation protocol are needed before an investment-performance claim.
