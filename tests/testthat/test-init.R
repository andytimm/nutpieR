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

test_that("init with partial list fills missing params randomly", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  # sigma supplied, mu missing -> mu filled randomly; should still sample OK.
  draws <- nutpie_sample(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 100, num_chains = 2, seed = 42, refresh = 0,
    init = list(sigma = 1)
  )
  expect_s3_class(draws, "draws_array")
})

test_that("init as list-of-lists provides per-chain starts", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_chains = 3, seed = 42, refresh = 0,
    init = list(
      list(mu = -2, sigma = 0.5),
      list(mu = 0, sigma = 1),
      list(mu = 2, sigma = 4)
    )
  )
  expect_equal(posterior::nchains(draws), 3)
})

test_that("init from JSON file path works", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))
  jsonlite::write_json(list(mu = 0, sigma = 1), tmp, auto_unbox = TRUE)

  draws <- nutpie_sample(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    init = tmp
  )
  expect_s3_class(draws, "draws_array")
})

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

test_that("init = 0 starts every chain at the origin and samples", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_chains = 2, seed = 42, refresh = 0,
    init = 0
  )
  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::nchains(draws), 2)
})

test_that("init = <positive scalar> samples successfully", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_chains = 2, seed = 42, refresh = 0,
    init = 2
  )
  expect_s3_class(draws, "draws_array")
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

test_that("init = function(chain_id) samples and yields distinct starts", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_chains = 3, seed = 42, refresh = 0,
    init = function(chain_id) list(mu = chain_id - 2, sigma = 1)
  )
  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::nchains(draws), 3)

  # Internal check: the dispatcher produces 3 distinct position vectors.
  handle <- nutpieR:::bs_open(
    test_models$normal$lib_path,
    nutpieR:::resolve_data(list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0))),
    0L
  )
  resolved <- nutpieR:::resolve_init(
    init = function(chain_id) list(mu = chain_id - 2, sigma = 1),
    init_mean = NULL, handle = handle, num_chains = 3, seed = 42
  )
  positions <- resolved$positions
  expect_length(positions, 3)
  mu_starts <- vapply(positions, `[`, numeric(1), 1L)
  expect_equal(sort(mu_starts), c(-1, 0, 1))
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
