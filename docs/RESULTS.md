# What the calculations mean

The original public report and saved notebook do not establish a profitable long-short strategy. This revision publishes code corrections and synthetic known-answer checks. It does not reissue historical financial outputs or disclose metadata or fingerprints from private Drive inputs. The old artifacts are explicitly superseded under ../archive/.

## Short-position accounting

Let L be the positive-score group's stock return and S the negative-score group's stock return if held long. The original negative short weights make short_return equal to -S. The original expression long_return - short_return therefore equals L + S. With one unit allocated to each group, it is equivalent to 200% long exposure before financing.

The intended spread under the original signed-weight convention is long_return + short_return, equal to L - S. The current module instead uses positive short magnitudes and subtracts the underlying shorted-stock payoff once. These conventions agree.

| Synthetic case | Original expression | Correct long-short P&L |
|---|---:|---:|
| Long stock +10%; shorted stock +20% | +30% | -10% |
| Long stock +10%; shorted stock -20% | -10% | +30% |
| Both stock groups +10% | +20% | 0% |

Tests execute exact archived original cells and repaired calculation cells on known-answer fixtures, and also exercise the current data and portfolio modules. A 72-month synthetic panel gives identical returns to all stocks within a month: a fully matched long-short portfolio must have zero spread. The corrected specifications satisfy that condition. This demonstrates accounting behavior, not historical investment success.

## Return metrics

- Average monthly return is the arithmetic mean of the monthly portfolio P&L divided by the stated capital.
- Cumulative return is prod(1+r) minus one.
- Annualized compounded return is prod(1+r) ** (12/T) minus one, for a valid complete monthly path and capital convention.
- Scaling exposure changes each monthly payoff; the scaled path must be compounded again. CAGR cannot simply be halved or doubled.
- Hit rate measures the fraction of positive months; it does not guarantee positive compounded performance.
- Drawdown includes starting capital. A zero-cash-baseline Sharpe is labeled explicitly. Benchmark-based IR uses aligned active returns and tracking error.

Positive absolute returns from buying both groups do not establish useful sentiment ranking. Any long-only comparison needs an equal-exposure benchmark and appropriate risk/cost assumptions. Dollar-neutral weights also do not by themselves establish beta neutrality or implementability.

## What remains necessary

Supply and independently validate dated issuer links, complete return coverage and missing/terminal payoff evidence. Use whole-record filing chronology and exact calendar targets, form holdings before checking future payoffs, and report unresolved held outcomes rather than inventing or discarding them. The supplied current ticker snapshot cannot certify historical identities.

Agree on security eligibility, filing/trade timing, capital, financing, borrow and trading costs, then precommit an untouched holdout or walk-forward protocol. Descriptive observed-outcome regressions are conditional associations; month effects contribute to full-model R-squared. Clustering improves uncertainty estimates under its assumptions but does not cure invalid links or selection.

The current public evidence supports the specified calculation corrections and safeguards. It does not establish a profitable strategy, underpricing, market-wide predictive power, out-of-sample performance, or that complexity caused worse returns. It also does not prove that all filing-sentiment models fail.
