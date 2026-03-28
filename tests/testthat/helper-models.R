# Compile test models once and cache for all tests.
# testthat sources helper-*.R before running any test file.

test_models <- new.env(parent = emptyenv())

try_compile <- function(name, ...) {
  tryCatch(
    nutpie_compile_model(...),
    error = function(e) NULL
  )
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
