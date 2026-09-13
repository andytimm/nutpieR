data { int<lower=1> n; real mu; real<lower=0> sigma; }
parameters { vector[n] x; }
model { x ~ normal(mu, sigma); }
