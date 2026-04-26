data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
transformed parameters {
  real mu_sq = mu * mu;
}
model {
  mu ~ normal(0, 10);
  sigma ~ cauchy(0, 5);
  y ~ normal(mu, sigma);
}
generated quantities {
  array[N] real y_rep;
  for (i in 1:N) y_rep[i] = normal_rng(mu, sigma);
}
