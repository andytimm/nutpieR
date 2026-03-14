test_that("sample_normal returns correct dimensions", {
  draws <- sample_normal(100L, 2L, 42L)
  expect_true(is.matrix(draws))
  expect_equal(nrow(draws), 200)
  expect_equal(ncol(draws), 10)
})

test_that("sample_normal means are near 0 and SDs near 1", {
  draws <- sample_normal(500L, 4L, 42L)
  col_means <- colMeans(draws)
  col_sds <- apply(draws, 2, sd)
  expect_true(all(abs(col_means) < 0.5))
  expect_true(all(abs(col_sds - 1) < 0.5))
})

test_that("resolve_data handles NULL", {
  expect_equal(nutpieR:::resolve_data(NULL), "")
})

test_that("resolve_data handles JSON string", {
  json <- '{"N": 10}'
  expect_equal(nutpieR:::resolve_data(json), json)
})

test_that("resolve_data handles list", {
  result <- nutpieR:::resolve_data(list(N = 10, y = c(0, 1)))
  parsed <- jsonlite::fromJSON(result)
  expect_equal(parsed$N, 10)
  expect_equal(parsed$y, c(0, 1))
})

test_that("resolve_data handles .json file", {
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))
  writeLines('{"N": 5}', tmp)
  result <- nutpieR:::resolve_data(tmp)
  expect_equal(jsonlite::fromJSON(result)$N, 5)
})

test_that("resolve_data rejects invalid input", {
  expect_error(nutpieR:::resolve_data(123))
})

test_that("nutpie_compile_model returns nutpie_model", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")
  model <- nutpie_compile_model(stan_file)
  expect_s3_class(model, "nutpie_model")
  expect_true(file.exists(model$lib_path))
})

test_that("nutpie_sample returns draws_array with correct dims", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 200, num_chains = 2, seed = 42
  )

  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::ndraws(draws), 400)
  expect_equal(posterior::nchains(draws), 2)
  expect_equal(posterior::niterations(draws), 200)
  expect_true("theta" %in% posterior::variables(draws))
})

test_that("nutpie_sample bernoulli theta is reasonable", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 500, num_chains = 4, seed = 123
  )

  summ <- posterior::summarize_draws(draws)
  theta_mean <- summ$mean[summ$variable == "theta"]
  # With 2/10 successes, posterior mean should be ~0.25
  expect_true(abs(theta_mean - 0.25) < 0.15)
})

test_that("nutpie_sample normal model param names correct", {
  stan_file <- test_path("test_models", "normal.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 200, num_chains = 2, seed = 42
  )

  expect_s3_class(draws, "draws_array")
  vars <- posterior::variables(draws)
  expect_true("mu" %in% vars)
  expect_true("sigma" %in% vars)
})
