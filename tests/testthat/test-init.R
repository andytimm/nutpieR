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
    nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init = list(mu = 0, sigma = 1),
      init_mean = 0
    ),
    "At most one"
  )
})

test_that("init and init_unconstrained are mutually exclusive", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init = list(mu = 0, sigma = 1),
      init_unconstrained = c(mu = 0, sigma = 0)
    ),
    "At most one"
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
