test_that("init_unconstrained with named vector samples successfully", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  unc <- nutpie_unconstrain(
    test_models$normal,
    list(mu = 0, sigma = 1),
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0))
  )
  draws <- nutpie_sample(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 100, num_chains = 2, seed = 42, refresh = 0,
    init_unconstrained = unc
  )
  expect_s3_class(draws, "draws_array")
})

test_that("init_unconstrained accepts per-chain list", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  unc_names <- nutpie_param_names(test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)))
  mk <- function(mu) stats::setNames(c(mu, 0), unc_names)

  draws <- nutpie_sample(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_chains = 3, seed = 42, refresh = 0,
    init_unconstrained = list(mk(-1), mk(0), mk(1))
  )
  expect_equal(posterior::nchains(draws), 3)
})

test_that("init_unconstrained errors on wrong-length vector", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init_unconstrained = c(mu = 0, sigma = 0, extra = 0)
    )
  )
})

test_that("init_unconstrained errors on wrong names", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init_unconstrained = c(alpha = 0, beta = 0)
    )
  )
})

test_that("init_unconstrained errors on unnamed vector", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_sample(
      test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init_unconstrained = c(0, 0)
    )
  )
})
