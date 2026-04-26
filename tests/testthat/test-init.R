# Helper: open a bs handle for the normal model (cached compile + small data).
# Used by the resolve_init unit tests below.
open_normal_handle <- function() {
  data_list <- list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0))
  nutpieR:::bs_open(
    test_models$normal$lib_path, nutpieR:::resolve_data(data_list), 0L
  )
}

# --- Integration tests (real sampler runs) ----------------------------------

test_that("init with full constrained list samples successfully", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 100, num_chains = 2, seed = 42, refresh = 0,
    init = list(mu = 0, sigma = 1)
  )
  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::niterations(draws), 100)
})

test_that("init = function(chain_id) samples and yields distinct starts", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  # Real sampler run verifies the function-form init plumbs through cleanly
  # (the per-chain dispatcher is checked directly below).
  draws <- nutpie_sample(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_chains = 3, seed = 42, refresh = 0,
    init = function(chain_id) list(mu = chain_id - 2, sigma = 1)
  )
  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::nchains(draws), 3)

  # Internal check: dispatcher produces 3 distinct position vectors.
  handle <- open_normal_handle()
  resolved <- nutpieR:::resolve_init(
    init = function(chain_id) list(mu = chain_id - 2, sigma = 1),
    init_mean = NULL, handle = handle, num_chains = 3, seed = 42
  )
  positions <- resolved$positions
  expect_length(positions, 3)
  mu_starts <- vapply(positions, `[`, numeric(1), 1L)
  expect_equal(sort(mu_starts), c(-1, 0, 1))
})

test_that("partial init is reproducible from sampler seed", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  data_list <- list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0))

  # Advance the global RNG between calls so that any reliance on it would
  # produce different random fills.
  draws1 <- nutpie_sample(
    test_models$normal, data = data_list,
    num_draws = 50, num_chains = 2, seed = 123, refresh = 0,
    init = list(sigma = 1)
  )
  invisible(stats::runif(10))
  draws2 <- nutpie_sample(
    test_models$normal, data = data_list,
    num_draws = 50, num_chains = 2, seed = 123, refresh = 0,
    init = list(sigma = 1)
  )
  expect_equal(as.array(draws1), as.array(draws2))
})

# --- Direct resolve_init unit tests (no sampler) ----------------------------

test_that("partial named-list init fills missing params per chain", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  handle <- open_normal_handle()

  # Only sigma is supplied; mu is missing and should be filled per chain.
  resolved <- nutpieR:::resolve_init(
    init = list(sigma = 1), init_mean = NULL,
    handle = handle, num_chains = 4, seed = 42
  )
  positions <- resolved$positions
  expect_length(positions, 4)
  mu_starts <- vapply(positions, `[`, numeric(1), 1L)
  # All chains' mu fills should be distinct (with probability ~1).
  expect_equal(length(unique(mu_starts)), 4L)
  # sigma slot (unconstrained = log(sigma) = log(1) = 0) should match per chain.
  sigma_unc <- vapply(positions, `[`, numeric(1), 2L)
  expect_true(all(abs(sigma_unc) < 1e-10))
})

test_that("fully-specified named-list init broadcasts a single position", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  handle <- open_normal_handle()
  resolved <- nutpieR:::resolve_init(
    init = list(mu = 0.5, sigma = 1), init_mean = NULL,
    handle = handle, num_chains = 4, seed = 42
  )
  # Length-1 = broadcast (Rust copies it to every chain).
  expect_length(resolved$positions, 1L)
})

test_that("list-of-lists init produces one position per chain", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  handle <- open_normal_handle()
  resolved <- nutpieR:::resolve_init(
    init = list(
      list(mu = -2, sigma = 0.5),
      list(mu = 0, sigma = 1),
      list(mu = 2, sigma = 4)
    ),
    init_mean = NULL, handle = handle, num_chains = 3, seed = 42
  )
  expect_length(resolved$positions, 3L)
  # Each chain's mu (unconstrained = constrained for unbounded mu) should
  # match the supplied value.
  mu_starts <- vapply(resolved$positions, `[`, numeric(1), 1L)
  expect_setequal(mu_starts, c(-2, 0, 2))
})

test_that("init from JSON file path parses and resolves", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))
  jsonlite::write_json(list(mu = 0, sigma = 1), tmp, auto_unbox = TRUE)

  handle <- open_normal_handle()
  resolved <- nutpieR:::resolve_init(
    init = tmp, init_mean = NULL,
    handle = handle, num_chains = 1, seed = 42
  )
  # Fully-specified -> broadcast (length 1).
  expect_length(resolved$positions, 1L)
  expect_length(resolved$positions[[1]], 2L)
})

test_that("init = 0 starts every chain at the origin", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  handle <- open_normal_handle()
  resolved <- nutpieR:::resolve_init(
    init = 0, init_mean = NULL,
    handle = handle, num_chains = 2, seed = 42
  )
  # Scalar 0 broadcasts a single all-zero position (Rust copies to every chain).
  expect_length(resolved$positions, 1L)
  expect_true(all(resolved$positions[[1]] == 0))
})

test_that("init = <positive scalar> draws Uniform(-x, x) per chain", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  handle <- open_normal_handle()
  resolved <- nutpieR:::resolve_init(
    init = 2, init_mean = NULL,
    handle = handle, num_chains = 3, seed = 42
  )
  expect_length(resolved$positions, 3L)
  # All draws within [-2, 2].
  all_vals <- unlist(resolved$positions)
  expect_true(all(all_vals >= -2 & all_vals <= 2))
})

test_that("partial init resolves even when generated quantities can't be evaluated", {
  # Regression: partial-init random fills used to call full param_constrain
  # (TP + GQ), so a model with bounded GQ that fails at random unconstrained
  # values would fail at init. The block-only constrain path avoids GQ.
  bad_gq_model <- tryCatch(
    nutpie_compile_model(code = "
      parameters {
        real mu;
        real<lower=0> sigma;
      }
      model { }
      generated quantities {
        int<lower=0,upper=0> bad = 1;
      }
    "),
    error = function(e) NULL
  )
  skip_if(is.null(bad_gq_model), "Bad-GQ model failed to compile")

  handle <- nutpieR:::bs_open(bad_gq_model$lib_path, "", 0L)
  expect_no_error(
    resolved <- nutpieR:::resolve_init(
      init = list(sigma = 1), init_mean = NULL,
      handle = handle, num_chains = 2, seed = 42
    )
  )
  expect_length(resolved$positions, 2L)
})

# --- Error-path tests -------------------------------------------------------

test_that("init errors on unknown parameter name", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init = list(nonexistent = 0)
    ),
    "Unknown parameter"
  )
})

test_that("init and init_mean are mutually exclusive", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    suppressWarnings(nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init = list(mu = 0, sigma = 1),
      init_mean = 0
    )),
    "Supply either"
  )
})

test_that("init with wrong per-chain list length errors", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  # 3 chains but only 2 init lists
  expect_error(
    nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 3, seed = 42, refresh = 0,
      init = list(
        list(mu = 0, sigma = 1),
        list(mu = 1, sigma = 2)
      )
    )
  )
})

test_that("init = <negative scalar> errors", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init = -1
    ),
    "non-negative"
  )
})

test_that("init function with zero args errors with clear message", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init = function() list(mu = 0, sigma = 1)
    ),
    "at least one argument"
  )
})

test_that("init function returning non-list errors", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init = function(chain_id) "not a list"
    ),
    "named list"
  )
})

# --- init_mean (soft-deprecated) --------------------------------------------

test_that("init_mean = scalar broadcasts to the unconstrained dim", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  handle <- open_normal_handle()  # ndim_unc = 2 (mu, log(sigma))
  expect_warning(
    resolved <- nutpieR:::resolve_init(
      init = NULL, init_mean = 0.5, handle = handle, num_chains = 2, seed = 42
    ),
    "init_mean.*deprecated"
  )
  expect_length(resolved$positions, 1L)  # broadcast
  expect_equal(resolved$positions[[1]], c(0.5, 0.5))
  expect_true(resolved$jitter)
})

test_that("init_mean = numeric vector of correct length passes through", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  handle <- open_normal_handle()
  expect_warning(
    resolved <- nutpieR:::resolve_init(
      init = NULL, init_mean = c(0.5, 1.0), handle = handle,
      num_chains = 2, seed = 42
    ),
    "init_mean.*deprecated"
  )
  expect_equal(resolved$positions[[1]], c(0.5, 1.0))
  expect_true(resolved$jitter)
})

test_that("init_mean of wrong length errors", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  handle <- open_normal_handle()
  expect_error(
    suppressWarnings(nutpieR:::resolve_init(
      init = NULL, init_mean = c(0.1, 0.2, 0.3), handle = handle,
      num_chains = 2, seed = 42
    )),
    "does not match model dimension"
  )
})
