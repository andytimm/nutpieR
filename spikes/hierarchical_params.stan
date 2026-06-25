// Larger, more complex model with many constrained params + GQ to maximize 
// R-side allocation in the progress callback.
// ~500K parameters, many transformed, to stress the allocator.
data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  vector[N] y;
}
parameters {
  vector[K] beta;
  real<lower=0> sigma;
  vector[N] alpha;
}
transformed parameters {
  vector[N] mu = alpha + X * beta;
}
model {
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  alpha ~ normal(0, 1);
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] y_rep;
  real log_lik = 0;
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu[n], sigma);
    log_lik += normal_lpdf(y[n] | mu[n], sigma);
  }
}
