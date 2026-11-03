# Configuration checks do not need a compiled Stan model.
adaptation_config <- function(...) {
  args <- list(
    seed = 1L, adaptation = "diag", low_rank_modified_mass_matrix = FALSE,
    progress = "none", save_warmup = FALSE, include = TRUE,
    store_divergences = FALSE, store_mass_matrix = FALSE,
    store_unconstrained = FALSE, store_gradient = FALSE,
    num_draws = 10L, num_warmup = NULL, num_chains = 1L, refresh = 0L,
    cores = 1L, max_treedepth = NULL, mindepth = NULL,
    extra_doublings = NULL, target_accept = NULL, max_energy_error = NULL,
    mass_matrix_gamma = NULL, mass_matrix_eigval_cutoff = NULL
  )
  do.call(nutpieR:::resolve_sample_config, modifyList(args, list(...)))
}

test_that("draw_diag is accepted without changing existing adaptation defaults", {
  expect_identical(adaptation_config()$adaptation, "diag")
  expect_identical(adaptation_config()$num_warmup, 400L)
  cfg <- adaptation_config(adaptation = "draw_diag")
  expect_identical(cfg$adaptation, "draw_diag")
  expect_identical(cfg$num_warmup, 400L)
  expect_identical(
    adaptation_config(adaptation = "draw_diag", num_warmup = 123L)$num_warmup,
    123L
  )
  expect_identical(adaptation_config(adaptation = "low_rank")$num_warmup, 800L)
  expect_identical(adaptation_config(adaptation = "low-rank")$adaptation, "low_rank")
  expect_error(adaptation_config(adaptation = "flow"), "arg")
  expect_error(adaptation_config(adaptation = "draw_diag", num_warmup = 0L),
               "num_warmup")
})

test_that("draw_diag ignores low-rank-only tuning and preserves legacy precedence", {
  cfg <- adaptation_config(adaptation = "draw_diag", mass_matrix_gamma = 0,
                           mass_matrix_eigval_cutoff = 0)
  expect_null(cfg$mass_matrix_gamma)
  expect_null(cfg$mass_matrix_eigval_cutoff)
  expect_warning(
    cfg <- adaptation_config(adaptation = "draw_diag",
                             low_rank_modified_mass_matrix = TRUE),
    "deprecated"
  )
  expect_identical(cfg$adaptation, "low_rank")
  expect_identical(cfg$num_warmup, 800L)
})

test_that("draw_diag changes only the diagonal estimator in native settings", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  common <- list(
    model = test_models$bernoulli, data = bernoulli_data(),
    num_draws = 20L, num_warmup = 100L, num_chains = 1L,
    cores = 1L, seed = 42L, progress = "none", save_warmup = TRUE,
    store_mass_matrix = TRUE, store_unconstrained = TRUE, store_gradient = TRUE,
    max_treedepth = 7L, target_accept = 0.9
  )
  diag <- do.call(nutpie_sample, c(common, list(adaptation = "diag")))
  draw_diag <- do.call(nutpie_sample, c(common, list(adaptation = "draw_diag")))
  diag_cfg <- jsonlite::fromJSON(attr(diag, "sampler_config"))
  draw_cfg <- jsonlite::fromJSON(attr(draw_diag, "sampler_config"))
  expect_true(diag_cfg$adapt_options$mass_matrix_options$use_grad_based_estimate)
  expect_false(draw_cfg$adapt_options$mass_matrix_options$use_grad_based_estimate)
  draw_cfg$adapt_options$mass_matrix_options$use_grad_based_estimate <- TRUE
  expect_equal(draw_cfg, diag_cfg)
  expect_equal(draw_cfg$num_warmup, 100)
  expect_equal(draw_cfg$maxdepth, 7)
  expect_equal(draw_cfg$adapt_options$step_size_settings$target_accept, 0.9)

  expect_s3_class(draw_diag, "draws_array")
  expect_equal(dim(draw_diag), c(20L, 1L, 1L))
  expect_true(all(is.finite(as.numeric(draw_diag))))
  expect_equal(dim(nutpie_warmup_draws(draw_diag)), c(100L, 1L, 1L))
  diagnostics <- nutpie_diagnostics(draw_diag)
  for (column in c("mass_matrix_inv", "unconstrained_draw", "gradient")) {
    expect_true(is.matrix(diagnostics[[column]]), info = column)
    expect_equal(dim(diagnostics[[column]]), c(20L, 1L), info = column)
    expect_true(all(is.finite(diagnostics[[column]])), info = column)
  }
  expect_true(all(diagnostics$mass_matrix_inv > 0))
})

test_that("draw_diag records its default warmup in the returned configuration", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  draws <- nutpie_sample(
    test_models$bernoulli, data = bernoulli_data(),
    adaptation = "draw_diag", num_draws = 10L, num_chains = 1L,
    cores = 1L, seed = 42L, progress = "none"
  )
  cfg <- jsonlite::fromJSON(attr(draws, "sampler_config"))
  expect_identical(attr(draws, "num_warmup"), 400L)
  expect_equal(cfg$num_warmup, 400)
  expect_false(cfg$adapt_options$mass_matrix_options$use_grad_based_estimate)
})
