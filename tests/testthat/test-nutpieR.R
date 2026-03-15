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
    num_draws = 200, num_chains = 2, seed = 42, refresh = 0
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
    num_draws = 500, num_chains = 4, seed = 123, refresh = 0
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
    num_draws = 200, num_chains = 2, seed = 42, refresh = 0
  )

  expect_s3_class(draws, "draws_array")
  vars <- posterior::variables(draws)
  expect_true("mu" %in% vars)
  expect_true("sigma" %in% vars)
})

test_that("nutpie_sample accepts sampling parameters", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 100, num_warmup = 200, num_chains = 2,
    max_treedepth = 8, target_accept = 0.9, seed = 42,
    refresh = 0
  )

  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::niterations(draws), 100)
  expect_equal(posterior::nchains(draws), 2)
})

test_that("nutpie_diagnostics returns expected fields", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 100, num_chains = 2, seed = 42, refresh = 0
  )

  diag <- nutpie_diagnostics(draws)
  expect_type(diag, "list")

  expected_fields <- c("diverging", "depth", "energy", "energy_error",
                       "logp", "n_steps", "step_size_bar", "mean_tree_accept")
  for (field in expected_fields) {
    expect_true(field %in% names(diag), info = paste("missing field:", field))
  }

  # Each diagnostic vector should have num_draws * num_chains elements
  expect_equal(length(diag$diverging), 200)
  expect_equal(length(diag$depth), 200)
  expect_equal(length(diag$logp), 200)

  # Sanity checks
  expect_type(diag$diverging, "logical")
  expect_type(diag$depth, "integer")
  expect_type(diag$energy, "double")
  expect_true(all(diag$depth > 0))
  expect_true(all(is.finite(diag$logp)))
  expect_true(all(diag$mean_tree_accept >= 0 & diag$mean_tree_accept <= 1))
})

test_that("nutpie_diagnostics errors on non-nutpie object", {
  expect_error(nutpie_diagnostics(1:10), "No diagnostics found")
})

# --- Item 1: Compile from Stan code string ---

test_that("nutpie_compile_model accepts code string", {
  code <- "data { int N; } parameters { real mu; } model { mu ~ normal(0, 1); }"
  model <- nutpie_compile_model(code = code)
  expect_s3_class(model, "nutpie_model")
  expect_true(file.exists(model$lib_path))
})

test_that("nutpie_compile_model rejects both stan_file and code", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")
  expect_error(
    nutpie_compile_model(stan_file = stan_file, code = "data {}"),
    "exactly one"
  )
})

test_that("nutpie_compile_model rejects neither stan_file nor code", {
  expect_error(nutpie_compile_model(), "exactly one")
})

# --- Item 4: Graceful error handling ---

test_that("sampling with bad data gives R error, not crash", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  expect_error(
    nutpie_sample(model, data = '{"bad": "json format"}',
                  num_draws = 10, num_chains = 1, seed = 42, refresh = 0)
  )
})

# --- Item 5: save_warmup ---

test_that("save_warmup returns warmup draws", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 100, num_warmup = 50, num_chains = 2, seed = 42,
    refresh = 0, save_warmup = TRUE
  )

  warmup <- nutpie_warmup_draws(draws)
  expect_s3_class(warmup, "draws_array")
  expect_equal(posterior::niterations(warmup), 50)
  expect_equal(posterior::nchains(warmup), 2)
  expect_true("theta" %in% posterior::variables(warmup))

  warmup_diag <- nutpie_warmup_diagnostics(draws)
  expect_type(warmup_diag, "list")
  expect_equal(length(warmup_diag$diverging), 100)  # 50 * 2 chains
})

test_that("nutpie_warmup_draws errors without save_warmup", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0
  )

  expect_error(nutpie_warmup_draws(draws), "save_warmup")
})

# --- Item 6: cores parameter ---

test_that("cores parameter works", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 50, num_chains = 4, cores = 2, seed = 42, refresh = 0
  )

  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::nchains(draws), 4)
})

# --- Item 7: store_divergences and store_mass_matrix ---

test_that("store_divergences adds divergence detail fields", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    store_divergences = TRUE
  )

  diag <- nutpie_diagnostics(draws)
  expect_true("divergence_start" %in% names(diag))
  expect_true("divergence_end" %in% names(diag))
  expect_true("divergence_momentum" %in% names(diag))
  expect_true("divergence_start_gradient" %in% names(diag))
  expect_type(diag$divergence_start, "list")
  expect_equal(length(diag$divergence_start), 50)
})

test_that("store_mass_matrix adds mass_matrix_inv field", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")

  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    store_mass_matrix = TRUE
  )

  diag <- nutpie_diagnostics(draws)
  expect_true("mass_matrix_inv" %in% names(diag))
  expect_type(diag$mass_matrix_inv, "list")
  expect_equal(length(diag$mass_matrix_inv), 50)
  # At least some entries should be non-NULL numeric vectors
  non_null <- Filter(Negate(is.null), diag$mass_matrix_inv)
  expect_true(length(non_null) > 0)
  expect_type(non_null[[1]], "double")
})
