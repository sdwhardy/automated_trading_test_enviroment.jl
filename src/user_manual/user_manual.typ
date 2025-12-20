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
The logarithmic return measures the relative change in an asset’s price
using the natural logarithm. It is time-additive and symmetric for gains and losses.

For a price series $P_t$, the $n$-period log-return at time $t$ is defined as

$ r_t = ln((P_t) / (P_(t - n))) $

where:
- $P_t$ is the price at time $t$
- $n$ is the return horizon and has the same units as $t$.

Log-returns are preferred in many statistical and optimization models
because cumulative returns over multiple periods can be obtained by
simple summation.

=== Signed Logarithmic Returns
The signed logarithmic return $(r_t^s)$ is defined as:

$ r_t^s = (r_t >= 0 ? 1 : -1) dot log(1 + abs(r_t)) $

Properties:
- Preserves the direction of the original return.
- Compresses large positive and negative returns symmetrically.
- Behaves approximately linearly near zero.
- Improves numerical stability for PCA and feature standardization.

=== Rate of Change (ROC)

The ROC measures the relative change in an asset's
price over $n$ periods. It is commonly used as a momentum indicator
in technical analysis.

For a price series $P_t$, the $n$-period ROC at time $t$ is defined as

$ R_t = (P_t - P_(t - n) ) / P_(t - n) $

ROC values indicate:
- Positive: price has increased over the period
- Negative: price has decreased over the period

=== Daily Realized Volatility

Daily realized volatility is a backward-looking measure of market
variability based on observed price movements. It is computed from
squared daily log-returns over a fixed rolling window.

Let $P_t$ denote the asset price at time $t$ and define the one-period
log-return as

$ r_t = ln(P_t / P_(t - 1)) $

The realized volatility at time $t$, using a window of $n$ periods, is given by

$ sigma_t = sqrt((1 / n) dot sum(r_(t-i)^2)), quad i = 1, dots, n $

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

The Garman–Klass volatility estimator is a range-based measure that
uses daily open, high, low, and close prices to estimate volatility
more efficiently than close-to-close methods.

Let $O_t$, $H_t$, $L_t$, and $C_t$ denote the open, high, low, and close
prices at time $t$.

The daily Garman–Klass variance contribution is defined as

$ v_t = 0.5 dot ln(H_t / L_t)^2 - (2 dot ln(2) - 1) dot ln(C_t / O_t)^2 $

Using a rolling window of $n$ days, the annualized volatility at time $t$
is given by

$ sigma_t = sqrt(A dot (1 / n) dot sum(v_{t-i})), quad i = 1, dots, n $

This estimator is unbiased under zero drift and is statistically more
efficient than close-to-close volatility when intraday price ranges
are informative.

=== Volatility of Volatility (VoV)

VoV measures how unstable or variable
market volatility itself is over time. Rather than focusing on price
returns, it quantifies fluctuations in realized volatility.

Let $sigma_t^d$ denote the daily realized volatility computed from prices.

Using a rolling window of $n$ days, define the average realized volatility as

$ m_t = (1 / n) dot sum(sigma_(t-1)^d), quad i = 0, dots, n - 1 $

The volatility of volatility is then defined as

$ v_t = sqrt((1 / n) dot sum((sigma_(t-1) - m_t)^2)), quad i = 0, dots, n - 1 $


A high value of $v_t$ indicates rapidly changing volatility and is often
associated with regime shifts, market stress, or heightened uncertainty.

Volatility of volatility is widely used in volatility modeling, risk
control, and derivative pricing applications.

=== Moving Average Difference

The moving average difference compares short-term and long-term price
trends by subtracting a long-term moving average from a short-term one.
It is a simple momentum and trend indicator.

Let $P_t$ denote the asset price at time $t$. Define the short-term and
long-term moving averages as

$ S_t = (1 / n_s) dot sum(P_(t-i)), quad i = 0, dots, n_s - 1 $

$ L_t = (1 / n_l) dot sum(P_(t-i)), quad i = 0, dots, n_l - 1 $

where $n_s < n_l$.

The moving average difference is given by

$ D_t = S_t - L_t $

Positive values of $D_t$ indicate that short-term prices exceed the
long-term trend, suggesting upward momentum, while negative values
indicate downward momentum.

=== Short-Term Price Slope

The short-term slope measures the local linear trend of prices over a rolling
window of $n$ observations.

For each time index $t >= n$, the slope is computed by fitting a straight line to
the prices $P_(t-n+1), dots, P_t$ using least squares.

Letting the time indices within the window be $i = 0, dots, n-1$.
The slope is defined as:

$
s_t =
  (sum_(i = 0)^(n - 1) (i - overline(i)) dot (P_(t - i) - overline(P)))
  /
  (sum_(i = 0)^(n - 1) (i - overline(i))^2)
$
Where:

$overline(i)$ is the average of the indices $0, ..., n - 1$

$overline(P)$ is the average price over the window

The resulting series captures short-term momentum:
positive values indicate upward trends, while negative values indicate downward
trends. This indicator is typically z-score standardized before use in PCA or
clustering.

=== Exponential Moving Average (EMA)

Let $P_t$ denote the price at time step $t$.
The exponential moving average with window length $n$ is defined recursively.

The smoothing factor is
$ alpha = 2 / (n + 1) $.

The EMA is initialized as
$ E_n = P_n $.

For all $t > n$, the EMA evolves according to
$ E_t = alpha · P_t + (1 - alpha) · E_{t-1} $.

Values for $t < n$ are undefined and treated as missing.

This formulation assigns exponentially decreasing weights to older observations,
giving greater importance to recent prices while maintaining smoothness.

=== Exponential Moving Average Difference

Let $P_t$ denote the price at time index $t$.
For two window lengths $n_s$ (short) and $n_l$ (long) with $n_s < n_l$,

The EMA difference feature is then defined as:

$ D_t = E_t^(n_s) - E_t^(n_l) $

If either EMA is undefined at time $t$, then $D_t$ is undefined.

This indicator measures short-term momentum relative to a longer-term
trend and is commonly used to detect trend direction and regime changes.

=== EMA Slope

Let $E_t$ denote the exponential moving average with window length $n$.
The EMA slope is defined as the first forward difference:

$ S_t = E_t - E_(t-1), quad t > n $

The normalized EMA slope feature is then defined as:

$ S_t = (E_t - E_(t-1)) / E_(t-1) $

Values for $t <= n$ are undefined.
This measures the instantaneous rate of change of the smoothed price series.

=== Volume Z-Score

The volume z-score measures how abnormal current trading volume is
relative to recent historical levels. It expresses volume deviations in
units of standard deviation, making volume spikes comparable across
assets and time.

Let $V_t$ denote trading volume at time $t$. Using a rolling window of
$n$ observations, define the mean and standard deviation as

$ mu_t = (1 / n) dot sum(V_(t-i)), quad i = 1, dots, n $

$ sigma_t = sqrt((1 / n) dot sum(V_(t-i)^2)), quad i = 1, dots, n $


The volume z-score is given by

$ Z_t = (V_t - mu_t) / sigma_t $

Large positive values of $Z_t$ indicate unusually high volume, often
associated with information arrival, breakouts, or regime changes.

=== Amihud Illiquidity Measure

The Amihud illiquidity measure quantifies market liquidity by relating
price movements to trading volume. It captures the idea that illiquid
assets experience larger price changes for a given amount of trading.

Let $P_t$ denote the asset price and $V_t$ the traded volume at time $t$.
Define the absolute return and dollar volume as

$ R_t = abs(P_t - P_(t-1)) / P_(t-1) $

$ D_t = P_t dot V_t $

The daily illiquidity contribution is given by

$ I_t = R_t / D_t $

Using a rolling window of $n$ observations, the Amihud illiquidity
measure is defined as

$ A_t = (1 / n) dot sum(I_(t-i)), quad i = 1, dots, n $

Higher values of $A_t$ indicate lower liquidity, as prices react more
strongly to a given level of trading activity.
== Normalization
=== Nonlinear Pre-Standardization Transforms

Before applying Z-score standardization, selected variables are transformed
to reduce skewness, stabilize variance, and limit the influence of extreme
values. This improves numerical stability and interpretability prior to
mean–variance scaling.

=== Signed Logarithmic Transform

For ROC, a signed logarithmic transform is applied.

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

=== Logarithmic Transform (Amihud Illiquidity)

The Amihud illiquidity measure is strictly non-negative and typically
right-skewed. A standard logarithmic compression is therefore applied.

Let $A_t$ denote the Amihud illiquidity value. The transformed value is

$ a = ln(1 + A_t) $

Adding 1 ensures the transform is well-defined at zero and avoids numerical
instability.

=== Interaction with Z-Score Standardization

These nonlinear transforms are applied prior to Z-score standardization.
Their purpose is not to enforce normality, but to produce distributions
that are better behaved under mean–variance scaling.

== Z-Score Standardization

Z-score standardization rescales data so that each variable has zero mean
and unit variance. This transformation makes variables comparable across
different scales and is commonly used in clustering, regression, and
dimensionality reduction.

Let $X_i$ denote the value of a variable $x$ at row $i$.

Let $mu_x$ denote the average of $X_i$ over the observed rows.

Let $sigma_x$ denote the standard deviation of $X_i$ over the observed rows.

The Z-score standardized value is then given by

$ Z_i = (X_i - mu_x) / sigma_x $

Missing values are preserved, and if $sigma_c$ is zero or undefined, it
is replaced by 1 to ensure numerical stability.

Z-score standardization removes scale effects while preserving the
relative structure of the data.

== Principle Component Analysis

PCA is used to transform a set of correlated features into an orthogonal
basis ordered by explained variance. In this pipeline, PCA is applied after
Z-score normalization.

=== 1. Missing-value filtering

Let $X$ denote a data matrix with $N$ rows and $d$ features.
Each row corresponds to one observation, and each column corresponds
to a feature.

Rows containing at least one missing feature value are removed prior
to principal component analysis.

Let $I$ denote the set of row indices for which all feature values are
observed (that is, rows with no missing entries).

The filtered data matrix is denoted by $X_I$.

Principal Component Analysis (PCA) is performed exclusively on $X_I$.

=== 2. Centering

Each feature is centered by subtracting its sample mean:

$ X_c = X - mu^T $

where

$ mu = (1 / N) · ∑_(i=1)^N X_i $

Even though features are already standardized, explicit centering ensures
numerical stability.

=== 3. Covariance matrix

The sample covariance matrix is computed as:

$ Sigma(j,k) = (1 / (N - 1)) · ∑_(i=1)^N X_c(i,j) · X_c(i,k) $

This captures the linear dependence structure between features.

=== 4. Eigen decomposition

The covariance matrix is decomposed as:

$ sum_(j=1)^d Sigma(i,j) · v_k(j) = lambda_k · v_k(i) $

where:
- $lambda_k$ are eigenvalues (explained variances),
- $v_k$ are orthonormal eigenvectors (principal directions).

Eigenvalues are sorted in descending order:

$ lambda_1 ≥ lambda_2 ≥ lambda_3 ≥ … ≥ lambda_d $

=== 5. Projection onto principal components

Principal component scores are obtained via:

$
Z = X_c V
$

where $ V = [v_1, v_2, dots, v_d] $.

Each column of $Z$ represents a principal component, with variance equal
to its corresponding eigenvalue.

=== 6. Properties

- Principal components are uncorrelated
- Total variance is preserved:

$ ∑_(k=1)^d lambda_k = tr(Sigma) $

- Reconstruction is possible via:


$ X_c ≈ Z · V^T $

== Clustering
=== Log-likelihood of a Gaussian Mixture Model

Let $X = { x_1, x_2, ..., x_n } subset RR^d$ be a dataset of n independent samples.
A Gaussian Mixture Model with K components is defined as

$
p(x) = sum_(k=1)^K pi_k dot cal(n)(x | mu_k, Sigma_k),
$

where the mixture weights satisfy

$
sum_(k=1)^K pi_k = 1.
$

The log-likelihood of the dataset under the model is

$
cal(L)(X)
= sum_(i=1)^n log(p(x_i))
= sum_(i=1)^n log(
  sum_(k=1)^K pi_k dot cal(n)(x_i | mu_k, Sigma_k)
).
$
=== Bayesian Information Criterion (BIC)
Assume a Gaussian Mixture Model with $k$ components and diagonal covariance
matrices.

The model density is defined as
$
p(x) = sum_(j=1)^k pi_j dot cal(n)(x | mu_j, Sigma_j),
$
where $pi_j >= 0$ and $sum_(j=1)^k pi_j = 1$.

The total log-likelihood of the dataset is
$
cal(L)(X) = sum_(i=1)^n log( p(x_i) ).
$

The average log-likelihood is calculated as:
$
overline(ell) = (1 / n) dot cal(L)(X),
$
so that
$
overline(cal(L)) = n dot overline(ell).
$

For a diagonal-covariance GMM, the number of free parameters is
$
p = (k dot d)      
  + (k dot d)      
  + (k - 1)    
$

- means contribute $k dot d$ parameters,
- variances contribute $k dot d$ parameters,
- mixture weights contribute $k - 1$ parameters.

The Bayesian Information Criterion (B) is then
$
B = -2 dot overline(cal(L)) + p dot log(n).
$

Lower BIC values indicate a better trade-off between model fit and complexity.

== Discussion
Interpret the results and implications.

== Conclusion
Summarize findings and next steps.

// ----------------------------
// References (optional)
// ----------------------------
== References

- I. T. Jolliffe, *Principal Component Analysis*, Springer  
  https://link.springer.com/book/10.1007/978-1-4757-1904-8
- Bishop, *Pattern Recognition and Machine Learning*  
  https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/

