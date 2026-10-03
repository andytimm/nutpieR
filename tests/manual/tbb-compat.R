# Run after installing nutpieR, once per fresh R process in both load orders:
# Rscript tests/manual/tbb-compat.R brms-first [path/to/RcppParallel-library]
# Rscript tests/manual/tbb-compat.R nutpie-first [path/to/RcppParallel-library]
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || !args[[1L]] %in% c("brms-first", "nutpie-first")) {
  stop("Expected brms-first or nutpie-first", call. = FALSE)
}
if (length(args) >= 2L) {
  .libPaths(c(args[[2L]], .libPaths()))
  actual <- normalizePath(find.package("RcppParallel"))
  expected <- normalizePath(args[[2L]])
  if (!identical(dirname(actual), expected) ||
      packageVersion("RcppParallel") != "6.2.1") {
    stop("Expected RcppParallel 6.2.1 in ", expected, "; got ", actual,
         call. = FALSE)
  }
}
if (args[[1L]] == "brms-first") library(brms)
library(nutpieR)
model <- nutpie_compile_model(code = "
  data { int<lower=0> N; array[N] int<lower=0,upper=1> y; }
  parameters { real<lower=0,upper=1> theta; }
  model { theta ~ beta(1, 3); y ~ bernoulli(theta); }
", cache = FALSE, verbose = 0L)
data <- list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1))
sample_one <- function(i) {
  draws <- nutpie_sample(model, data, num_draws = 10L, num_warmup = 10L,
                         num_chains = 2L, cores = 2L, seed = 604L,
                         refresh = 0L)
  stopifnot(dim(draws)[[1L]] == 10L)
  invisible(NULL)
}
sample_one(1L)
if (args[[1L]] == "nutpie-first") library(brms)
if (requireNamespace("furrr", quietly = TRUE)) {
  future::plan(future::multisession, workers = 2L)
  furrr::future_map(1:2, sample_one)
  future::plan(future::sequential)
}
gc()
cat("TBB_COMPAT_SUCCESS", args[[1L]], "RcppParallel",
    as.character(packageVersion("RcppParallel")), "\n")
