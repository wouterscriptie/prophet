// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

functions {
  real[ , ] get_changepoint_matrix(real[] t, real[] t_change, int T, int S) {
    // Assumes t and t_change are sorted.
    real A[T, S];
    real a_row[S];
    int cp_idx;

    // Start with an empty matrix.
    A = rep_array(0, T, S);
    a_row = rep_array(0, S);
    cp_idx = 1;

    // Fill in each row of A.
    for (i in 1:T) {
      while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
        a_row[cp_idx] = 1;
        cp_idx = cp_idx + 1;
      }
      A[i] = a_row;
    }
    return A;
  }

  // Linear trend function

  real[ , ] linear_trend(
    real[] k,
    real[] m,
    real[ , ] delta,
    real[] t,
    real[ , ] A,
    real[] t_change,
    int S,
    int T,
    int F
  ) {
    real gamma[S, F];
    real Y[T, F];
    for (f in 1:F)
      for (i in 1:S) {
        gamma[i,f] = -t_change[i] * delta[i,f];
    }

    for (f in 1:F)
      for (i in 1:S) {
        Y[i,f] = (k[f] + dot_product(A[i], delta[ , f])) * t[i] + (
          m[f] + dot_product(A[i], gamma[ , f]));
    }
    return Y;
  }
}

data {
  int T;                // Number of time periods
  int F;                // Amount of different time series 
  int<lower=1> K;       // Number of regressors
  real t[T];            // Time
  real cap[T];          // Capacities for logistic trend
  real y[T, F];            // Time series
  int S;                // Number of changepoints
  real t_change[S];     // Times of trend changepoints
  real X[T,K];         // Regressors
  vector[K] sigmas;     // Scale on seasonality prior
  real<lower=0> tau;    // Scale on changepoints prior
  int trend_indicator;  // 0 for linear, 1 for logistic, 2 for flat
  real s_a[K];          // Indicator of additive features
  real s_m[K];          // Indicator of multiplicative features
}

transformed data {
  real A[T, S];
  A = get_changepoint_matrix(t, t_change, T, S);
}

parameters {
  real k[F];                   // Base trend growth rate
  real m[F];                   // Trend offset
  real delta[S, F];            // Trend rate adjustments
  real<lower=0> sigma_obs[F];  // Observation noise
  real beta[K, F];             // Regressor coefficients
  //hyper parameters
  real mu_beta[K];
  real<lower=0> sigma_beta[K];
  real mu_delta[S];
  real<lower=0> sigma_delta[S];
}

transformed parameters {
  real trend[T, F];
  real Y[T, F];
  real beta_m[K, F];
  real beta_a[K, F];
  print("test", Y)

  if (trend_indicator == 0) {
    trend = linear_trend(k, m, delta, t, A, t_change, S, T, F);
  }

  for (f in 1:F)   
    for (i in 1:K) {
      beta_m[i,f] = beta[i,f] * s_m[i];
      beta_a[i,f] = beta[i,f] * s_a[i];
  }

  for (f in 1:F)
    for (i in 1:T){
      Y[i,f] = (
        trend[i,f] * (1 + dot_product(X[i], beta_m[ , f])) + dot_product(X[i], beta_a[ , f])
    );
  }
}

model {
  //hyperpriors
  //mu_beta ~ normal(0, sigmas);
  //sigma_beta ~ normal(0, 0.01);
  //mu_delta ~ double_exponential(0, tau);
  //sigma_delta ~ normal(0, 0.01);
  //priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  sigma_obs ~ normal(0, 0.5);
  for (f in 1:F){
    //delta[ , f] ~ normal(mu_delta, sigma_delta);
    delta[ , f] ~ double_exponential(0, tau);
  }
  for (f in 1:F){  
    //beta[ , f] ~ normal(mu_beta, sigma_beta); 
    beta[ , f] ~ normal(0, sigmas); 
  }
  // Likelihood
  for (f in 1:F){
    y[ , f] ~ normal(Y[ , f], sigma_obs[f]);
  }
}
