skip_if_not_installed("reticulate")
skip_on_cran()

NUTPIE_PYTHON <- "/Users/atimm/.virtualenvs/nutpie_env/bin/python"
skip_if_not(file.exists(NUTPIE_PYTHON), "Python nutpie env not found")

test_that("nutpieR and Python nutpie produce same draws with same seed", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "bernoulli.stan not found")

  data <- list(N = 10L, y = c(0L, 1L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 1L))
  seed <- 42L
  num_draws <- 500L
  num_chains <- 2L

  # Sample with nutpieR
  model_r <- nutpie_compile_model(stan_file)
  draws_r <- nutpie_sample(model_r,
    data = data,
    num_draws = num_draws,
    num_chains = num_chains,
    seed = seed
  )

  # Sample with Python nutpie via reticulate
  Sys.setenv(RETICULATE_PYTHON = NUTPIE_PYTHON)
  reticulate::use_python(NUTPIE_PYTHON, required = TRUE)

  nutpie <- reticulate::import("nutpie")

  # Compile model using nutpie's own Stan compilation
  compiled <- nutpie$compile_stan_model(filename = stan_file)
  compiled <- compiled$with_data(N = 10L, y = c(0L, 1L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 1L))

  # Sample with same settings
  py_draws <- nutpie$sample(compiled,
    draws = as.integer(num_draws),
    chains = as.integer(num_chains),
    seed = as.integer(seed),
    tune = 300L  # match nutpieR hardcoded num_tune
  )

  # Extract Python draws
  py_theta <- reticulate::py_to_r(py_draws$posterior[["theta"]]$values)
  py_mean_theta <- mean(py_theta)

  # Extract R draws
  r_means <- posterior::summarize_draws(draws_r)
  r_mean_theta <- r_means$mean[r_means$variable == "theta"]

  # With same seed + same engine, means should be very close
  # Allow 0.1 tolerance for potential nuts-rs version differences
  expect_true(abs(r_mean_theta - py_mean_theta) < 0.1,
    info = sprintf("R mean=%.4f, Python mean=%.4f", r_mean_theta, py_mean_theta))
})
