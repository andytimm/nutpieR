data { int<lower=1> n; real mu; real<lower=0> sigma; }
parameters { vector<lower=0>[n] x; }
transformed parameters { vector[n] square_x = square(x); }
model { x ~ normal(mu, sigma); }
generated quantities { real predictive = normal_rng(mu, sigma); }
