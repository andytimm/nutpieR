test_that("nutpie_param_names returns unconstrained names", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  names_unc <- nutpie_param_names(
    test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    unconstrained = TRUE
  )
  expect_type(names_unc, "character")
  expect_equal(names_unc, "theta")
})

test_that("nutpie_param_names returns constrained names (incl TP/GQ)", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  names_con <- nutpie_param_names(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    unconstrained = FALSE
  )
  expect_true("mu" %in% names_con)
  expect_true("sigma" %in% names_con)
})

test_that("nutpie_param_names works for unconstrained normal model", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  names_unc <- nutpie_param_names(
    test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    unconstrained = TRUE
  )
  expect_length(names_unc, 2)
  expect_true(all(c("mu", "sigma") %in% names_unc))
})

test_that("nutpie_unconstrain round-trips identity params", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  # sigma = 1 -> log(sigma) = 0 in unconstrained space
  unc <- nutpie_unconstrain(
    test_models$normal,
    list(mu = 0, sigma = 1),
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0))
  )
  expect_type(unc, "double")
  expect_length(unc, 2)
  expect_false(is.null(names(unc)))
  # mu is identity; sigma has lower=0 -> log transform: log(1) = 0
  expect_equal(unname(unc[["mu"]]), 0, tolerance = 1e-10)
  expect_equal(unname(unc[["sigma"]]), 0, tolerance = 1e-10)
})

test_that("nutpie_unconstrain applies log transform to bounded params", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  # sigma = exp(2) -> unconstrained = 2
  unc <- nutpie_unconstrain(
    test_models$normal,
    list(mu = 3, sigma = exp(2)),
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0))
  )
  expect_equal(unname(unc[["mu"]]), 3, tolerance = 1e-10)
  expect_equal(unname(unc[["sigma"]]), 2, tolerance = 1e-10)
})

test_that("nutpie_unconstrain errors on wrong-named params", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_unconstrain(
      test_models$normal,
      list(mu = 0, nonexistent = 1),
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0))
    )
  )
})

test_that("nutpie_unconstrain errors on unnamed list", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_unconstrain(test_models$normal, list(0, 1),
                       data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0))),
    "fully named"
  )
})
