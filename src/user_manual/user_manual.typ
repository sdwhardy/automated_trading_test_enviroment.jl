// ----------------------------
// Document metadata
// ----------------------------
#set page(
  paper: "a4",
  margin: 2cm,
)

#set text(
  font: "Latin Modern Roman",
  size: 11pt,
)

// ----------------------------
// Title
// ----------------------------
= User Manual  
#datetime.today()

// ----------------------------
// Abstract (optional)
// ----------------------------
== Abstract
This document briefly describes the purpose of the work.

// ----------------------------
// Main sections
// ----------------------------
== Introduction
Introduce the problem, background, and motivation.

== Indicators

=== Logarithmic Returns
The *logarithmic return* measures the relative change in an asset’s price
using the natural logarithm. It is widely used in quantitative finance
because it is time-additive and symmetric for gains and losses.

For a price series $P_t$, the $n$-period log-return at time $t$ is defined as

$ r_t = ln((P_t) / (P_(t - n))) $

where:
- $P_t$ is the price at time $t$
- $n$ is the return horizon in periods

Log-returns are preferred in many statistical and optimization models
because cumulative returns over multiple periods can be obtained by
simple summation.

=== Rate of Change (ROC)

The *Rate of Change* (ROC) measures the relative change in an asset's
price over $n$ periods. It is commonly used as a momentum indicator
in technical analysis.

For a price series $P_t$, the $n$-period ROC at time $t$ is defined as

$ R_t = (P_t - P_{t-n}) / P_{t-n} $

where:
- $P_t$ is the price at time $t$
- $n$ is the lookback period

ROC values indicate:
- Positive: price has increased over the period
- Negative: price has decreased over the period

=== Daily Realized Volatility

*Daily realized volatility* is a backward-looking measure of market
variability based on observed price movements. It is computed from
squared daily log-returns over a fixed rolling window.

Let $P_t$ denote the asset price at time $t$ and define the one-period
log-return as

$ r_t = ln(P_t / P_{t-1}) $

The realized volatility at time $t$, using a window of $n$ days, is given by

$ sigma_t = sqrt((1 / n) dot sum(r_{t-i}^2)), quad i = 1, dots, n $

This estimator captures the magnitude of recent price fluctuations and
is widely used as a proxy for true (latent) volatility in empirical
finance.

=== Annualized Realized Volatility

Annualized realized volatility rescales daily realized volatility to a
yearly horizon under the assumption that daily returns are independent
and identically distributed.

Let $sigma_t^d$ denote the daily realized volatility computed over a
rolling window of $n$ days. The annualized realized volatility is given by

$ sigma_t = sqrt(A) dot sigma_t^d $

where $A$ is the number of trading periods per year, typically $A = 252$.

This transformation enables direct comparison of volatility estimates
across assets and time horizons.

=== Garman–Klass Volatility

The *Garman–Klass volatility estimator* is a range-based measure that
uses daily open, high, low, and close prices to estimate volatility
more efficiently than close-to-close methods.

Let $O_t$, $H_t$, $L_t$, and $C_t$ denote the open, high, low, and close
prices at time $t$.

The daily Garman–Klass variance contribution is defined as

$ v_t = 0.5 dot ln(H_t / L_t)^2 - (2 dot ln(2) - 1) dot ln(C_t / O_t)^2 $

Using a rolling window of $n$ days, the annualized volatility at time $t$
is given by

$ sigma_t = sqrt(A dot (1 / n) dot sum(v_{t-i})), quad i = 1, dots, n $

where $A$ is the annualization factor (typically 252 trading days).

This estimator is unbiased under zero drift and is statistically more
efficient than close-to-close volatility when intraday price ranges
are informative.

=== Volatility of Volatility

*Volatility of volatility* (VoV) measures how unstable or variable
market volatility itself is over time. Rather than focusing on price
returns, it quantifies fluctuations in realized volatility.

Let $sigma_t^d$ denote the daily realized volatility computed from prices.
Using a rolling window of $m$ days, the volatility of volatility is defined as

$ v_t = std(sigma_{t-i}^d), quad i = 0, dots, m - 1 $

A high value of $v_t$ indicates rapidly changing volatility and is often
associated with regime shifts, market stress, or heightened uncertainty.

Volatility of volatility is widely used in volatility modeling, risk
control, and derivative pricing applications.

=== Moving Average Difference

The *moving average difference* compares short-term and long-term price
trends by subtracting a long-term moving average from a short-term one.
It is a simple momentum and trend indicator widely used in technical
analysis.

Let $P_t$ denote the asset price at time $t$. Define the short-term and
long-term moving averages as

$ S_t = (1 / n_s) dot sum(P_{t-i}), quad i = 0, dots, n_s - 1 $

$ L_t = (1 / n_l) dot sum(P_{t-i}), quad i = 0, dots, n_l - 1 $

where $n_s < n_l$.

The moving average difference is given by

$ D_t = S_t - L_t $

Positive values of $D_t$ indicate that short-term prices exceed the
long-term trend, suggesting upward momentum, while negative values
indicate downward momentum.

=== Volume Z-Score

The *volume z-score* measures how abnormal current trading volume is
relative to recent historical levels. It expresses volume deviations in
units of standard deviation, making volume spikes comparable across
assets and time.

Let $V_t$ denote trading volume at time $t$. Using a rolling window of
$n$ observations, define the mean and standard deviation as

$ mu_t = (1 / n) dot sum(V_{t-i}), quad i = 1, dots, n $

$ sigma_t = std(V_{t-i}), quad i = 1, dots, n $

The volume z-score is given by

$ Z_t = (V_t - mu_t) / sigma_t $

Large positive values of $Z_t$ indicate unusually high volume, often
associated with information arrival, breakouts, or regime changes.

=== Amihud Illiquidity Measure

The *Amihud illiquidity measure* quantifies market liquidity by relating
price movements to trading volume. It captures the idea that illiquid
assets experience larger price changes for a given amount of trading.

Let $P_t$ denote the asset price and $V_t$ the traded volume at time $t$.
Define the absolute return and dollar volume as

$ R_t = abs(P_t - P_{t-1}) / P_{t-1} $

$ D_t = P_t dot V_t $

The daily illiquidity contribution is given by

$ I_t = R_t / D_t $

Using a rolling window of $n$ observations, the Amihud illiquidity
measure is defined as

$ A_t = (1 / n) dot sum(I_{t-i}), quad i = 1, dots, n $

Higher values of $A_t$ indicate lower liquidity, as prices react more
strongly to a given level of trading activity.
== Normalization
=== Nonlinear Pre-Standardization Transforms

Before applying Z-score standardization, selected variables are transformed
to reduce skewness, stabilize variance, and limit the influence of extreme
values. This improves numerical stability and interpretability prior to
mean–variance scaling.

=== Signed Logarithmic Transform (ROC)

For rate-of-change features, a signed logarithmic transform is applied.

Let $x$ denote a return value.

Define $s(x)$ as the sign of $x$, taking values $1$ for positive $x$,
$-1$ for negative $x$, and $0$ otherwise.

The signed logarithmic transform is defined as

$ f(x) = s(x) dot ln(1 + |x|) $

This transform has the following properties:

- Preserves the sign of the original return  
- Compresses extreme positive and negative values symmetrically  
- Behaves approximately linearly near zero  

It is well suited for financial return series, which often exhibit heavy
tails and outliers.

In the implementation, this transform is applied to:

- 21-day rate of change  
- 63-day rate of change  

Let $r_k$ denote a rate-of-change feature at horizon $k$. The transformed
feature is given by

$ r_k = s(r_k) dot ln(1 + abs(r_k)) $

---

=== Logarithmic Transform (Amihud Illiquidity)

The Amihud illiquidity measure is strictly non-negative and typically
right-skewed. A standard logarithmic compression is therefore applied.

Let $a$ denote the Amihud illiquidity value. The transformed value is

$ a = ln(1 + a) $

Adding 1 ensures the transform is well-defined at zero and avoids numerical
instability.

---

=== Interaction with Z-Score Standardization

These nonlinear transforms are applied *prior* to Z-score standardization.
Their purpose is not to enforce normality, but to produce distributions
that are better behaved under mean–variance scaling.

After transformation, Z-score standardization is applied as follows.

Let $X_i$ denote an observation of a variable $x$, with average $mu_x$
and standard deviation $sigma_x$. The standardized value is

$ Z_i = (X_i - mu_x) / sigma_x $

---

This two-step process:

- Reduces skewness and tail risk (log and signed-log transforms)  
- Removes scale and location effects (Z-score standardization)  

The result is a set of features that are more comparable across time and
variables, while preserving relative structure and sign information.

=== Z-Score Standardization

Z-score standardization rescales data so that each variable has zero mean
and unit variance. This transformation makes variables comparable across
different scales and is commonly used in clustering, regression, and
dimensionality reduction.

Let $X_i$ denote the value of a variable $x$ at row $i$.

Let $mu_x$ denote the average of $X_i$ over the observed rows.

Let $sigma_x$ denote the standard deviation of $X_i$ over the observed rows.

The standardized value is then given by

$ Z_i = (X_i - mu_x) / sigma_x $

Missing values are preserved, and if $sigma_c$ is zero or undefined, it
is replaced by 1 to ensure numerical stability.

Z-score standardization removes scale effects while preserving the
relative structure of the data.

== Principle Component Analysis
Details of a specific part.

== Results
Present results, figures, or tables.

== Discussion
Interpret the results and implications.

== Conclusion
Summarize findings and next steps.

// ----------------------------
// References (optional)
// ----------------------------
== References
- Reference 1
- Reference 2
