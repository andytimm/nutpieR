data { int<lower=1> n; real mu; real<lower=0> sigma; }
parameters { vector[n] y; }
model { y ~ normal(mu, sigma); }
