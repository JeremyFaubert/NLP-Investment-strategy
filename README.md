# SEC Filing Sentiment and Equity Returns

**Authors:** Jeremy Faubert and Matthias Baker

**Course:** FT370 — Investment Applications of NLP

**Tools:** Python, pandas, NumPy, statsmodels, matplotlib

Does the language in corporate annual and quarterly filings relate to the following month's stock returns? This project compares three sentiment-based portfolio approaches using precomputed Loughran–McDonald SEC filing summaries and monthly equity returns.

**Research status:** the current version corrects accounting and data-handling problems found during review. The earlier profitability and out-of-sample conclusions are withdrawn. Historical performance remains unvalidated pending dated security-to-issuer links, complete return coverage, and resolution of missing held returns. See [the correction notice](docs/CORRECTIONS.md).

## Research approach

The project normalizes filing word counts, compares firms within each month, forms portfolios, and evaluates the following month's returns. The configured formation period is January 1994–December 2023, with nominal holding months February 1994–January 2024.

| Approach | Signal | Portfolio construction in the current version |
|---|---|---|
| Baseline | Positive minus negative word counts, divided by uncertainty, then plus one | Long positive scores; short negative scores; weights proportional to score magnitude |
| Expanded | A combination of seven standardized language measures | Long the highest monthly quintile; short the lowest; equal weights within each side |
| Enhanced | Expanded signal with extreme formation-time values winsorized | Rank within industries; match industry allocations across both sides; use signal-strength weights with a hard 5% capital-based position cap |

The baseline retains `(Positive - Negative) / Uncertainty + 1` for comparison with the original specification. `(Positive - Negative) / (Uncertainty + 1)` is a separate, configurable specification; the parentheses change the signal.

For the expanded signal, each category count is divided by total filing words, then standardized across firms in the same month. The combination is:

```text
z(Positive) - z(Negative) - z(Uncertainty) - z(Litigious)
            + z(Strong Modal) - z(Weak Modal) - z(Constraining)
```

These signs are modeling choices, not evidence that a category predicts returns.

## What the seven language categories mean

The project uses category counts supplied in the [Loughran–McDonald filing summaries](https://sraf.nd.edu/sec-edgar-data/). It does not extract raw filing text or perform dictionary matching itself. The [official dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) defines the seven categories:

| Category | Language captured |
|---|---|
| Positive | Favorable or optimistic tone |
| Negative | Adverse or pessimistic tone |
| Uncertainty | Doubt, ambiguity, or unpredictability |
| Litigious | Legal matters and litigation |
| Strong modal | Certainty or strong commitment |
| Weak modal | Possibility or tentative statements |
| Constraining | Restrictions, limitations, or obligations |

## Why the accounting correction matters

The original baseline subtracted an already negative short-position payoff. It therefore added the returns of two long groups, equivalent to 200% long exposure under the stated capital convention before financing. The corrected calculation subtracts the shorted stocks' underlying returns exactly once.

| Synthetic example | Original expression | Correct long-short P&L |
|---|---:|---:|
| Long stock +10%; shorted stock +20% | +30% | −10% |
| Long stock +10%; shorted stock −20% | −10% | +30% |

Correct accounting can increase or decrease the result. A positive output alone does not demonstrate useful sentiment prediction. The enhanced implementation also previously clipped future realized returns; the current version winsorizes available signals and preserves actual payoffs. [Calculation interpretation](docs/RESULTS.md) explains the return measures and remaining limitations.

## Data handling and evaluation

- Pair returns with exactly the next calendar month before filtering; preserve missing and nonnumeric outcomes.
- Select one whole latest filing record and retain its accession number. Apply a three-calendar-month age limit and a disclosed one Monday–Friday business-day availability lag. This approximates timing when acceptance timestamps are absent; it is not an exchange trading calendar.
- Form holdings before checking whether future payoffs are available. An unresolved held return makes that portfolio month unavailable.
- Match enhanced industry allocations and enforce the position cap. Infeasible capacity stays unallocated. Each side can use at most one capital unit, allowing up to 200% gross exposure; unallocated capacity contributes zero excess P&L.
- Withhold full-period performance statistics while identifier, held-payoff, or benchmark checks fail. Descriptive regressions use month effects and issuer/month clustered uncertainty; they do not establish causality or out-of-sample prediction.

Default `legacy_diagnostic` mode uses a [frozen public SEC current-ticker snapshot](resources/README.md) for inspection. Current tickers cannot certify historical identities. Alternative `verified_links` mode requires an independently reviewed dated PERMNO–CIK table; the mode name does not certify its provenance or coverage. The CIK–CUSIP input cannot repair returns that lack historical CUSIP.

Trading costs, borrow availability, fees, and financing are excluded. Complete data validation and a precommitted holdout or walk-forward evaluation remain necessary before making an investment-performance claim. The evidence does not establish profitability, underpricing, or that added complexity caused worse performance.

## Run the public calculation checks

Python 3.12 was used for the recorded checks. Clone this repository, change to its root directory, then run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\activate`.

All **30 focused checks** use public synthetic fixtures and require no private financial inputs. They cover short accounting, calendar gaps, filing chronology, missing-outcome invariance, position caps, industry allocations, and metric definitions. Some execute labeled intermediate calculation fixtures; others exercise the current data and portfolio modules. Passing them verifies those behaviors, not the validity of a historical dataset.

## Run the research notebook

The [root notebook](NLP_Sentiment_Investing_model.ipynb) is the current entry point. Raw inputs are not redistributed; follow [data/README.md](data/README.md) to supply appropriately licensed source files. Set `NLP_DATA_DIR` and `NLP_OUTPUT_DIR` if needed; defaults are `data/` and `research_outputs/`. Keep source provenance and generated financial outputs private.

With the environment above activated, install and start JupyterLab from the repository root:

```bash
python -m pip install jupyterlab
jupyter lab NLP_Sentiment_Investing_model.ipynb
```

In Colab, clone the repository, change directory to it, and configure the data path; enable the optional Drive mount only if needed. Local code-cell execution was checked; Colab mounting/frontend integration was not tested. Dense month-effect regressions can require several GB of RAM; set `RUN_PANEL_REGRESSIONS=False` to inspect data and portfolio diagnostics without those fits.

Notebook outputs are cleared to prevent stale results from appearing beside corrected code. The original public notebook, report, and figures remain in [archive/](archive/README.md), clearly marked as superseded.

## Repository guide

| Path | Purpose |
|---|---|
| `NLP_Sentiment_Investing_model.ipynb` | Current research pipeline |
| `review_data.py`, `review_portfolios.py` | Data preparation and portfolio construction |
| `review_fixes.py`, `review_inference.py` | Accounting, metrics, and descriptive regressions |
| `tests/` | Synthetic checks and labeled calculation fixtures |
| `docs/` | Detailed corrections and calculation interpretation |
| `data/`, `resources/`, `results/` | Input instructions, public snapshot, and research-status notice |
| `archive/` | Superseded original public artifacts |
