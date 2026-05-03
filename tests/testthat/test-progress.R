test_that("progress = 'text' prints per-chain log lines", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  capture_messages(
    capture.output(
      draws <- nutpie_sample(
        test_models$bernoulli, data = bernoulli_data(),
        num_draws = 50, num_warmup = 50, num_chains = 1,
        seed = 1L, refresh = 10, progress = "text"
      ),
      type = "output"
    )
  )
  expect_s3_class(draws, "draws_array")
})

test_that("progress = 'progressr' fires progressor signals", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  skip_if_not_installed("progressr")
  signal_count <- 0L
  withr::with_options(
    list(progressr.enable = TRUE),
    {
      progressr::with_progress(
        handlers = progressr::handler_void(),
        {
          withCallingHandlers(
            draws <- nutpie_sample(
              test_models$bernoulli, data = bernoulli_data(),
              num_draws = 30, num_warmup = 30, num_chains = 2,
              seed = 1L, refresh = 1L, progress = "progressr"
            ),
            progression = function(cnd) {
              signal_count <<- signal_count + 1L
              invokeRestart("muffleProgression")
            }
          )
        }
      )
    }
  )
  expect_s3_class(draws, "draws_array")
  expect_gt(signal_count, 0L)
})

test_that("progress = 'none' produces no console output", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  expect_silent(
    draws <- nutpie_sample(
      test_models$bernoulli, data = bernoulli_data(),
      num_draws = 30, num_warmup = 30, num_chains = 1,
      seed = 1L, progress = "none"
    )
  )
  expect_s3_class(draws, "draws_array")
})

test_that("progress = 'auto' resolves to 'text' when non-interactive", {
  testthat::local_mocked_bindings(
    interactive = function() FALSE,
    .package = "base"
  )
  expect_equal(nutpieR:::resolve_progress_mode("auto", refresh = 100L), "text")
})

test_that("progress = 'auto' uses progressr when available + interactive", {
  testthat::local_mocked_bindings(
    interactive = function() TRUE,
    .package = "base"
  )
  skip_if_not_installed("progressr")
  skip_if_not_installed("cli")
  expect_equal(nutpieR:::resolve_progress_mode("auto", refresh = 100L), "progressr")
})

test_that("refresh = 0 wins over progress argument", {
  expect_equal(nutpieR:::resolve_progress_mode("progressr", refresh = 0L), "none")
  expect_equal(nutpieR:::resolve_progress_mode("auto", refresh = 0L), "none")
  expect_equal(nutpieR:::resolve_progress_mode("text", refresh = 0L), "none")
})

test_that("a buggy progress callback does not crash the sample", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  skip_if_not_installed("progressr")
  testthat::local_mocked_bindings(
    make_progressr_callback = function(num_chains, num_warmup, num_draws) {
      function(snapshot) stop("kaboom")
    },
    .package = "nutpieR"
  )
  testthat::local_mocked_bindings(
    interactive = function() TRUE,
    .package = "base"
  )
  capture.output(
    draws <- nutpie_sample(
      test_models$bernoulli, data = bernoulli_data(),
      num_draws = 30, num_warmup = 30, num_chains = 1,
      seed = 1L, refresh = 1L, progress = "progressr"
    ),
    type = "output"
  )
  expect_s3_class(draws, "draws_array")
})
