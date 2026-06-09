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
  expect_equal(summary$min_step_size, 0.015)
  expect_equal(summary$max_latest_num_steps, 63L)
  expect_equal(summary$first_divergence, "chain 2 draw 17")
  # status is "! div: 2 | <grad>" — divergence indicator present
  expect_match(summary$status, "div: 2", fixed = TRUE)
  # avg_num_steps = (900+1800)/(120+90) = 2700/210 ~= 12.86
  expect_match(summary$status, "grad/draw", fixed = TRUE)
})

test_that("format_status_tokens replaces div and grad tokens", {
  snapshot <- list(
    list(
      chain = 1L, finished_draws = 408L, total_draws = 2000L,
      divergences = 3L, tuning = FALSE, started = TRUE,
      latest_num_steps = 7L, total_num_steps = 1500L,
      step_size = 0.3, runtime = 12.0,
      divergent_draws = c(10L, 20L, 30L)
    ),
    list(
      chain = 2L, finished_draws = 465L, total_draws = 2000L,
      divergences = 0L, tuning = FALSE, started = TRUE,
      latest_num_steps = 3L, total_num_steps = 1800L,
      step_size = 0.4, runtime = 11.5,
      divergent_draws = integer()
    )
  )
  summary <- nutpieR:::summarize_progress_snapshot(snapshot)

  result <- nutpieR:::format_status_tokens(snapshot, summary, 10L, "{div} | {grad}")
  expect_match(result, "div:", fixed = TRUE)
  expect_match(result, "grad/draw", fixed = TRUE)

  # div-only format
  div_only <- nutpieR:::format_status_tokens(snapshot, summary, 10L, "{div}")
  expect_match(div_only, "div: 3", fixed = TRUE)
  expect_false(grepl("grad", div_only))
})

test_that("format_status_tokens handles empty tokens without leaving stray pipes", {
  snapshot <- list(
    list(
      chain = 1L, finished_draws = 100L, total_draws = 2000L,
      divergences = 0L, tuning = TRUE, started = TRUE,
      latest_num_steps = 3L, total_num_steps = 300L,
      step_size = 0.5, runtime = 5.0,
      divergent_draws = integer()
    ),
    list(
      chain = 2L, finished_draws = 100L, total_draws = 2000L,
      divergences = 0L, tuning = TRUE, started = TRUE,
      latest_num_steps = 3L, total_num_steps = 300L,
      step_size = 0.5, runtime = 5.0,
      divergent_draws = integer()
    )
  )
  summary <- nutpieR:::summarize_progress_snapshot(snapshot)

  # {lag} returns "" when chains are in sync — no stray pipes
  result <- nutpieR:::format_status_tokens(snapshot, summary, 10L, "{div} | {lag} | {grad}")
  expect_false(grepl("\\|\\s*\\|", result))
  expect_false(grepl("^\\s*\\|", result))
  expect_false(grepl("\\|\\s*$", result))
})

test_that("format_chain_draw_range produces range for multi-chain snapshots", {
  snapshot <- list(
    list(finished_draws = 408L, total_draws = 2000L),
    list(finished_draws = 465L, total_draws = 2000L),
    list(finished_draws = 430L, total_draws = 2000L)
  )
  result <- nutpieR:::format_chain_draw_range(snapshot)
  expect_match(result, "408", fixed = TRUE)
  expect_match(result, "465", fixed = TRUE)
  expect_match(result, "2k", fixed = TRUE)
  expect_match(result, "–", fixed = TRUE)  # en-dash

  # when all chains have same count, no dash
  snapshot_equal <- list(
    list(finished_draws = 500L, total_draws = 1000L),
    list(finished_draws = 500L, total_draws = 1000L)
  )
  result_eq <- nutpieR:::format_chain_draw_range(snapshot_equal)
  expect_false(grepl("–", result_eq))
  expect_match(result_eq, "1k", fixed = TRUE)
})

test_that("format_gradient_status uses triangle symbol when treedepth cap hit", {
  # avg_lf / max_possible >= 0.05 triggers warning
  # max_possible for max_treedepth=10 is 2^10 - 1 = 1023
  # 0.05 * 1023 = 51.15, so avg_lf >= 52 triggers warning
  result_warn <- nutpieR:::format_gradient_status(100.0, max_treedepth = 10L)
  expect_match(result_warn, "▲", fixed = TRUE)  # ▲
  expect_false(grepl("~", result_warn))

  result_ok <- nutpieR:::format_gradient_status(2.0, max_treedepth = 10L)
  expect_false(grepl("▲", result_ok))
  expect_false(grepl("~", result_ok))
})

test_that("make_text_progress_callback prints one line per chain at refresh interval", {
  output_lines <- character(0)
  local_cat <- function(..., sep = " ") {
    output_lines <<- c(output_lines, paste(..., sep = sep))
  }

  snapshot <- list(
    list(
      chain = 1L, finished_draws = 100L, total_draws = 1000L,
      divergences = 0L, tuning = TRUE, started = TRUE,
      latest_num_steps = 3L, total_num_steps = 300L,
      step_size = 0.5, runtime = 5.0,
      divergent_draws = integer()
    ),
    list(
      chain = 2L, finished_draws = 100L, total_draws = 1000L,
      divergences = 1L, tuning = TRUE, started = TRUE,
      latest_num_steps = 3L, total_num_steps = 300L,
      step_size = 0.5, runtime = 5.0,
      divergent_draws = c(50L)
    )
  )

  callback <- nutpieR:::make_text_progress_callback(
    num_chains = 2L, num_warmup = 400L, num_draws = 1000L,
    max_treedepth = 10L, refresh = 50L
  )
  # Capture cat output
  out <- capture.output(callback(snapshot), type = "output")

  # Should print 2 lines (one per chain), since finished=100 >= refresh=50
  expect_length(out, 2L)
  expect_match(out[1], "c1", fixed = TRUE)
  expect_match(out[2], "c2", fixed = TRUE)
  expect_match(out[1], "warmup", fixed = TRUE)
  expect_match(out[2], "div:", fixed = TRUE)

  # Second call with same snapshot should not print (since_last = 0 < 50)
  out2 <- capture.output(callback(snapshot), type = "output")
  expect_length(out2, 0L)
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
  expect_equal(updates[[1]]$extra$phase, "warmup")
})

test_that("explicit progressr progress samples successfully", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  skip_if_not_installed("progressr")
  progressr::handlers("void")
  capture.output(
    capture_messages(
      draws <- nutpie_sample(
        test_models$bernoulli, data = bernoulli_data(),
        num_draws = 30, num_warmup = 30, num_chains = 2,
        seed = 1L, refresh = 1L, progress = "progressr"
      )
    ),
    type = "output"
  )
  expect_s3_class(draws, "draws_array")
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

test_that("explicit text progress samples successfully", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  capture.output(
    capture_messages(
      draws <- nutpie_sample(
        test_models$bernoulli, data = bernoulli_data(),
        num_draws = 30, num_warmup = 30, num_chains = 2,
        seed = 1L, refresh = 10L, progress = "text"
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
