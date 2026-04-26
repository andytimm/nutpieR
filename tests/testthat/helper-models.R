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
