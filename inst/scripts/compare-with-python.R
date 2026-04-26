# Cross-implementation sanity check: nutpieR vs Python `nutpie` on the same
# model + data + seed. Run manually when bumping `nuts-rs` or `nutpie` (Python)
# versions to confirm the two wrappers still produce equivalent draws.
#
# Not part of the test suite. Requires reticulate + a Python env with `nutpie`
# installed. Set RETICULATE_PYTHON to point at that env, or rely on
# `reticulate::py_discover_config()` to find one.

if (!requireNamespace("reticulate", quietly = TRUE)) {
  stop("reticulate is required to run this script.", call. = FALSE)
}

python <- Sys.getenv("RETICULATE_PYTHON", unset = NA_character_)
if (is.na(python) || !nzchar(python)) {
  cfg <- tryCatch(reticulate::py_discover_config(), error = function(e) NULL)
  if (is.null(cfg) || !nzchar(cfg$python)) {
    stop("Could not resolve a Python interpreter. Set RETICULATE_PYTHON or ",
         "configure reticulate.", call. = FALSE)
  }
  python <- cfg$python
}
if (!file.exists(python)) {
  stop("Python interpreter not found at: ", python, call. = FALSE)
}
reticulate::use_python(python, required = TRUE)

stan_file <- system.file("test_models", "bernoulli.stan", package = "nutpieR",
                         mustWork = FALSE)
if (!nzchar(stan_file) || !file.exists(stan_file)) {
  # Fall back to the test fixture for in-tree runs
  stan_file <- file.path("tests", "testthat", "test_models", "bernoulli.stan")
}
if (!file.exists(stan_file)) {
  stop("bernoulli.stan not found.", call. = FALSE)
}

data <- list(N = 10L, y = c(0L, 1L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 1L))
seed <- 42L
num_draws <- 500L
num_warmup <- 400L  # nutpie_sample()'s current default
num_chains <- 2L

model_r <- nutpieR::nutpie_compile_model(stan_file)
draws_r <- nutpieR::nutpie_sample(
  model_r, data = data,
  num_draws = num_draws, num_warmup = num_warmup,
  num_chains = num_chains, seed = seed
)

nutpie <- reticulate::import("nutpie")
compiled <- nutpie$compile_stan_model(filename = stan_file)
compiled <- compiled$with_data(N = data$N, y = data$y)
py_draws <- nutpie$sample(
  compiled,
  draws = as.integer(num_draws),
  tune = as.integer(num_warmup),
  chains = as.integer(num_chains),
  seed = as.integer(seed)
)

py_theta <- reticulate::py_to_r(py_draws$posterior[["theta"]]$values)
py_mean_theta <- mean(py_theta)

r_means <- posterior::summarize_draws(draws_r)
r_mean_theta <- r_means$mean[r_means$variable == "theta"]

cat(sprintf("R mean(theta)      = %.4f\n", r_mean_theta))
cat(sprintf("Python mean(theta) = %.4f\n", py_mean_theta))
cat(sprintf("|delta|            = %.4f\n", abs(r_mean_theta - py_mean_theta)))

if (abs(r_mean_theta - py_mean_theta) >= 0.1) {
  warning("Means differ by >= 0.1 — investigate before bumping versions.",
          call. = FALSE)
}
