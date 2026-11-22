test_that("WALNUTS sampling preserves results and records effective controls", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")
  for (adaptation in c("diag", "draw_diag", "low_rank")) {
    for (controls in list(TRUE, list(energy_criterion = "endpoint",
                                   target_macro_steps = 15,
                                   max_leapfrog_steps = 500))) {
      fit <- nutpie_sample(test_models$normal, data = normal_data(),
        num_draws = 100, num_warmup = 200, num_chains = 3, cores = 2,
        seed = 813, adaptation = adaptation, walnuts = controls,
        save_warmup = TRUE, store_gradient = TRUE, progress = "none")
      expect_equal(dim(fit), c(100L, 3L, 2L))
      expect_true(all(is.finite(fit)))
      cfg <- jsonlite::fromJSON(attr(fit, "sampler_config"))
      expect_equal(cfg$gradient_clipping, 1e10)
      expect_equal(cfg$walnuts$max_step_halvings, 8)
      diagnostics <- nutpie_diagnostics(fit)
      expect_true(all(c("irreversible", "min_micro_steps", "n_gradients") %in% names(diagnostics)))
      expect_equal(nrow(diagnostics$gradient), 300L)
      expect_equal(dim(nutpie_warmup_draws(fit)), c(200L, 3L, 2L))
      expect_true(all(diagnostics$n_gradients >= diagnostics$n_steps))
      if (is.list(controls)) {
        expect_equal(cfg$walnuts$energy_criterion, "Endpoint")
        expect_equal(cfg$walnuts$target_macro_steps, 15)
        expect_equal(cfg$max_leapfrog_steps, 500)
        expect_true(all(c("macro_steps_mean", "min_micro_steps_unrounded") %in% names(diagnostics)))
      } else {
        expect_equal(cfg$walnuts$energy_criterion, "MaxMin")
        expect_null(cfg$walnuts$target_macro_steps)
      }
    }
  }
})

test_that("ordinary NUTS remains the default with clipping enabled", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")
  fit <- nutpie_sample(test_models$normal, data = normal_data(),
    num_draws = 30, num_warmup = 100, num_chains = 1, seed = 813,
    progress = "none")
  cfg <- jsonlite::fromJSON(attr(fit, "sampler_config"))
  expect_null(cfg$walnuts)
  expect_equal(cfg$gradient_clipping, 1e10)
  expect_null(nutpie_diagnostics(fit)$gradient)
  expect_null(nutpie_diagnostics(fit)$unconstrained_draw)
  expect_null(nutpie_diagnostics(fit)$irreversible)
})
