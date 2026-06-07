test_that("progress mode resolves to cli in interactive auto mode", {
  testthat::local_mocked_bindings(
    interactive = function() TRUE,
    .package = "base"
  )
  skip_if_not_installed("cli")

  expect_equal(nutpieR:::resolve_progress_mode("auto", refresh = 100L), "cli")
  expect_equal(nutpieR:::resolve_progress_mode("auto", refresh = 0L), "none")
  expect_equal(nutpieR:::resolve_progress_mode("none", refresh = 100L), "none")
})

test_that("progress mode falls back to text outside interactive sessions", {
  testthat::local_mocked_bindings(
    interactive = function() FALSE,
    .package = "base"
  )

  expect_equal(nutpieR:::resolve_progress_mode("auto", refresh = 100L), "text")
})

test_that("progress snapshot summary exposes useful sampler diagnostics", {
  snapshot <- list(
    list(
      chain = 1L, finished_draws = 120L, total_draws = 200L,
      divergences = 0L, tuning = TRUE, started = TRUE,
      latest_num_steps = 15L, total_num_steps = 900L,
      step_size = 0.12, runtime = 3.2,
      divergent_draws = integer()
    ),
    list(
      chain = 2L, finished_draws = 90L, total_draws = 200L,
      divergences = 2L, tuning = FALSE, started = TRUE,
      latest_num_steps = 63L, total_num_steps = 1800L,
      step_size = 0.015, runtime = 4.1,
      divergent_draws = c(17L, 88L)
    )
  )

  summary <- nutpieR:::summarize_progress_snapshot(snapshot)

  expect_equal(summary$total_finished, 210)
  expect_equal(summary$total_draws, 400)
  expect_equal(summary$phase, "mixed")
  expect_equal(summary$total_divergences, 2L)
  expect_equal(summary$slowest_chain, 2L)
  expect_equal(summary$min_step_size, 0.015)
  expect_equal(summary$max_latest_num_steps, 63L)
  expect_equal(summary$first_divergence, "chain 2 draw 17")
  expect_match(summary$status, "! div: 2 c2", fixed = TRUE)
  expect_match(summary$status, "chains: c2 behind 90/200")
  expect_match(summary$status, "grad: c2 63")
  expect_false(grepl("step", summary$status))
})

test_that("cli callback only advances by new draws", {
  updates <- list()
  fake_update <- function(set = NULL, status = NULL, extra = NULL, id = NULL, force = FALSE) {
    updates[[length(updates) + 1L]] <<- list(set = set, status = status, extra = extra, force = force)
  }
  callback <- nutpieR:::make_cli_progress_callback(
    num_chains = 2L, num_warmup = 10L, num_draws = 10L,
    id = "fake", update = fake_update, done = function(...) TRUE
  )

  callback(list(
    list(chain = 1L, finished_draws = 3L, total_draws = 20L, divergences = 0L, tuning = TRUE, started = TRUE, latest_num_steps = 2L, total_num_steps = 6L, step_size = 0.1, runtime = 0.1, divergent_draws = integer()),
    list(chain = 2L, finished_draws = 2L, total_draws = 20L, divergences = 0L, tuning = TRUE, started = TRUE, latest_num_steps = 3L, total_num_steps = 6L, step_size = 0.1, runtime = 0.1, divergent_draws = integer())
  ))
  callback(list(
    list(chain = 1L, finished_draws = 3L, total_draws = 20L, divergences = 0L, tuning = TRUE, started = TRUE, latest_num_steps = 2L, total_num_steps = 6L, step_size = 0.1, runtime = 0.1, divergent_draws = integer()),
    list(chain = 2L, finished_draws = 2L, total_draws = 20L, divergences = 0L, tuning = TRUE, started = TRUE, latest_num_steps = 3L, total_num_steps = 6L, step_size = 0.1, runtime = 0.1, divergent_draws = integer())
  ))

  expect_length(updates, 1L)
  expect_equal(updates[[1]]$set, 5)
  expect_equal(updates[[1]]$extra$phase, "warm")
})


test_that("explicit cli progress samples successfully", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  skip_if_not_installed("cli")
  capture.output(
    capture_messages(
      draws <- nutpie_sample(
        test_models$bernoulli, data = bernoulli_data(),
        num_draws = 30, num_warmup = 30, num_chains = 2,
        seed = 1L, refresh = 1L, progress = "cli"
      )
    ),
    type = "output"
  )
  expect_s3_class(draws, "draws_array")
})

test_that("progress = 'none' produces no console output", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  expect_silent(
    draws <- nutpie_sample(
      test_models$bernoulli, data = bernoulli_data(),
      num_draws = 30, num_warmup = 30, num_chains = 1,
      seed = 1L, refresh = 100L, progress = "none"
    )
  )
  expect_s3_class(draws, "draws_array")
})

test_that("a failing progress callback is disabled instead of killing sampling", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  call_count <- 0L
  testthat::local_mocked_bindings(
    make_cli_progress_callback = function(...) {
      function(snapshot) {
        call_count <<- call_count + 1L
        stop("kaboom")
      }
    },
    .package = "nutpieR"
  )
  capture.output(
    draws <- nutpie_sample(
      test_models$bernoulli, data = bernoulli_data(),
      num_draws = 30, num_warmup = 30, num_chains = 1,
      seed = 1L, refresh = 1L, progress = "cli"
    ),
    type = "output"
  )
  expect_s3_class(draws, "draws_array")
  expect_lte(call_count, 2L)
})
