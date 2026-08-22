parameters {
  real x;
}
model {
  x ~ normal(0, 1);
}
generated quantities {
  real<lower=0> invalid_gq = -1;
}
