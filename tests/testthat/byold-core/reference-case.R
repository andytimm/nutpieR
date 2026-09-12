# Run in a bounded child after installing the release package.
library(nutpieR)
example <- source(system.file("examples/byold/run.R", package = "nutpieR"))$value
reference <- nutpie_compile_model(code = "
  data { int n; real mu; real sigma; }
  transformed data { real centre = normal_rng(0, 10); }
  parameters { vector[n] x; }
  model { x ~ normal(centre, sigma); }
  generated quantities {
    real centre_out = centre;
    real noise = normal_rng(0, 1);
  }")
# Recover the seed-0 reference centre to supply matching evaluator data.
ptr <- nutpieR:::bs_open(reference$lib_path, '{"n":2,"mu":0,"sigma":2}', 0L)
centre <- nutpieR:::bs_evaluate(ptr, list(c(0, 0)))[[1]]$gradient[1] * 4
bound <- nutpie_attach_density_evaluator(reference, example$bound$evaluator_path,
                             list(n = 2L, mu = centre, sigma = 2))
stopifnot(nutpie_validate_density_evaluator(bound)$status == "pass")
sample <- function(seed) nutpie_sample(bound, num_draws = 100,
  num_warmup = 200, num_chains = 2, seed = seed, progress = "none")
a <- sample(42)
b <- sample(42)
c <- sample(43)
forward <- function(model, data = NULL, ...) nutpie_sample(model, data = data, ...)
forwarded <- forward(bound, num_draws = 100, num_warmup = 200,
                     num_chains = 2, seed = 42, progress = "none")
stopifnot(identical(as.numeric(a), as.numeric(forwarded)))
e_override <- tryCatch(forward(bound, data = list()), error = identity)
stopifnot(inherits(e_override, "error"), grepl("rebind", conditionMessage(e_override)))
stopifnot(all(abs(a[, , "centre_out"] - centre) < 1e-12),
          all(abs(c[, , "centre_out"] - centre) < 1e-12),
          identical(as.numeric(a), as.numeric(b)),
          !identical(as.numeric(a[, , "noise"]), as.numeric(c[, , "noise"])))
# Reusing the reference must not dereference a deserialized external pointer.
dead <- unserialize(serialize(bound, NULL))
e <- tryCatch(nutpie_sample(dead, num_draws = 2, num_warmup = 2,
                           progress = "none"), error = identity)
stopifnot(inherits(e, "error"), grepl("rebind", conditionMessage(e)))
cat("BOUND REFERENCE REGRESSIONS PASSED\n")
