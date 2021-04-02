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

  matrix linear_trend(
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
    real Y[T,F];
    real Y1[T, F];
    real Y2[T, F];
    matrix[S,F] delta_matrix;
    matrix[T,S] A_matrix;
    matrix[T,F] Y1_matrix;
    matrix[S,F] gamma_matrix;
    matrix[T,F] Y2_matrix;
    matrix[T,F] Y_matrix;
	  
    for (f in 1:F)
      for (i in 1:S) {
        gamma[i,f] = -t_change[i] * delta[i,f];
    }
    gamma_matrix = to_matrix(gamma);
    delta_matrix = to_matrix(delta);
    A_matrix = to_matrix(A);  
    Y1_matrix = (A_matrix * delta_matrix);
    Y1 = to_array_2d(Y1_matrix);
    for (f in 1:F)
      for (i in 1:T) {
        Y1[i,f] = (k[f] + Y1[i,f]) * t[i];
    }
    Y1_matrix = to_matrix(Y1);
    Y2_matrix = (A_matrix * gamma_matrix);
    Y2 = to_array_2d(Y2_matrix);
    for (f in 1:F)
      for (i in 1:T) {
        Y2[i,f] = (m[f] + Y2[i,f]);
    }
    Y2_matrix = to_matrix(Y2);
    Y_matrix = Y1_matrix + Y2_matrix;
    Y = to_array_2d(Y_matrix);
    return Y_matrix;
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
  real missing_values[T,F]; // indicator of missing values
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
  matrix[T,F] trend;
  matrix[T,F] Y_matrix_2;
  real Y[T, F];
  real beta_a[K, F];
  matrix[K,F] beta_matrix;
  matrix[T,K] X_matrix;
  matrix[T,F] seas_matrix; 
  real A_test[T,S];

  if (trend_indicator == 0) {
    trend = linear_trend(k, m, delta, t, A, t_change, S, T, F);
  }
  
  for (f in 1:F)   
    for (i in 1:K) {
      beta_a[i,f] = beta[i,f] * s_a[i];
  }
  beta_matrix = to_matrix(beta_a);
  X_matrix = to_matrix(X);
  seas_matrix = X_matrix * beta_matrix;
  
  Y_matrix_2 = trend + seas_matrix;
  Y = to_array_2d(Y_matrix_2);

  //for (f in 1:F)
    //for (i in 1:T){
      //Y[i,f] = (trend[i,f]);
   //}
}

model {
  //hyperpriors
  mu_beta ~ normal(0, sigmas);
  sigma_beta ~ normal(0, 0.01);
  mu_delta ~ double_exponential(0, tau);
  sigma_delta ~ normal(0, 0.01);

  //priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  sigma_obs ~ normal(0, 0.5);
  for (f in 1:F){
    delta[ , f] ~ normal(mu_delta, sigma_delta);
    //delta[ , f] ~ double_exponential(0, tau);
  }
  for (f in 1:F){  
    beta[ , f] ~ normal(mu_beta, sigma_beta); 
    //beta[ , f] ~ normal(0, sigmas); 
  }
  // Likelihood
  for (f in 1:F){
    y[ , f] ~ normal(Y[ , f], sigma_obs[f]);
  }
}
