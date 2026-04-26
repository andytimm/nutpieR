test_that("nutpie_param_names returns block-level names by default", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  names_block <- nutpie_param_names(
    test_models$bernoulli,
    data = bernoulli_data()
  )
  expect_type(names_block, "character")
  expect_equal(names_block, "theta")
})

test_that("nutpie_param_names which = 'block' returns just block names", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  names_block <- nutpie_param_names(
    test_models$normal,
    data = normal_data(),
    which = "block"
  )
  expect_equal(sort(names_block), c("mu", "sigma"))
})

test_that("nutpie_param_names which = 'unconstrained' returns unc names", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  names_unc <- nutpie_param_names(
    test_models$normal,
    data = normal_data(),
    which = "unconstrained"
  )
  expect_length(names_unc, 2)
  expect_true(all(c("mu", "sigma") %in% names_unc))
})

test_that("nutpie_param_names which = 'full' includes TP/GQ", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  names_full <- nutpie_param_names(
    test_models$normal,
    data = normal_data(),
    which = "full"
  )
  expect_true("mu" %in% names_full)
  expect_true("sigma" %in% names_full)
})

test_that("nutpie_param_names `unconstrained` is deprecated for both values", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  data <- normal_data()

  expect_warning(
    names_unc <- nutpie_param_names(test_models$normal, data = data,
                                    unconstrained = TRUE),
    "unconstrained.*deprecated"
  )
  expect_length(names_unc, 2)
  expect_true(all(c("mu", "sigma") %in% names_unc))

  expect_warning(
    names_full <- nutpie_param_names(test_models$normal, data = data,
                                     unconstrained = FALSE),
    "unconstrained.*deprecated"
  )
  expect_true("mu" %in% names_full)
  expect_true("sigma" %in% names_full)
})

test_that("nutpie_unconstrain round-trips identity params", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  # sigma = 1 -> log(sigma) = 0 in unconstrained space
  unc <- nutpie_unconstrain(
    test_models$normal,
    list(mu = 0, sigma = 1),
    data = normal_data()
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
    data = normal_data()
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
      data = normal_data()
    )
  )
})

test_that("nutpie_unconstrain errors on unnamed list", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_unconstrain(test_models$normal, list(0, 1),
                       data = normal_data()),
    "fully named"
  )
})
