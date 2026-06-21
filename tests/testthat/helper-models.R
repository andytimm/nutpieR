# Compile test models once and cache for all tests.
# testthat sources helper-*.R before running any test file.

test_models <- new.env(parent = emptyenv())

# Interactive devs without a Rust toolchain see a soft skip; CI / non-interactive
# runs propagate compile errors so a green run actually proves the models built.
require_models <- !interactive() || nzchar(Sys.getenv("CI"))

try_compile <- function(name, ...) {
  if (require_models) {
    return(nutpie_compile_model(...))
  }
  tryCatch(nutpie_compile_model(...), error = function(e) NULL)
}

test_models$bernoulli <- try_compile(
  "bernoulli",
  stan_file = test_path("test_models", "bernoulli.stan")
)

test_models$normal <- try_compile(
  "normal",
  stan_file = test_path("test_models", "normal.stan")
)

test_models$code_string <- try_compile(
  "code_string",
  code = "data { int N; } parameters { real mu; } model { mu ~ normal(0, 1); }"
)

test_models$tp_gq <- try_compile(
  "tp_gq",
  stan_file = test_path("test_models", "tp_gq.stan")
)

# Correlated 2D normal whose unconstrained space IS the parameter vector, so
# `nutpie_sample()` and `nutpie_sample_r()` sample the same density on the same
# scale — the basis for the R-callback cross-check in test-r-density.R.
test_models$mvn <- try_compile(
  "mvn",
  code = paste(
    "data { vector[2] mu; matrix[2,2] Sigma; }",
    "parameters { vector[2] y; }",
    "model { y ~ multi_normal(mu, Sigma); }",
    sep = "\n"
  )
)

# Shared fixtures used by test-init.R, test-helpers.R, test-nutpieR.R.

mvn_mu <- c(1, -2)
mvn_sigma <- matrix(c(1, 0.8, 0.8, 1), 2)

bernoulli_data <- function() {
  list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1))
}

normal_data <- function() {
  list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0))
}

open_normal_handle <- function() {
  nutpieR:::bs_open(
    test_models$normal$lib_path, nutpieR:::resolve_data(normal_data()), 0L
  )
}
