test_that("progress mode resolves to cli in interactive auto mode", {
  testthat::local_mocked_bindings(
    interactive = function() TRUE,
    .package = "base"
  )
  withr::local_options(knitr.in.progress = NULL)

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

test_that("auto mode falls back to text when rendering a knitr document", {
  testthat::local_mocked_bindings(
    interactive = function() TRUE,
    .package = "base"
  )
  withr::local_options(knitr.in.progress = TRUE)

  expect_equal(nutpieR:::resolve_progress_mode("auto", refresh = 100L), "text")
})

test_that("chain_format must be a scalar string with mode-specific tokens", {
  expect_null(nutpieR:::validate_chain_format(NULL, "cli"))
  expect_equal(
    nutpieR:::validate_chain_format("{div} | {grad}", "cli"),
    "{div} | {grad}"
  )
  expect_equal(
    nutpieR:::validate_chain_format("{div} | {tdepth}", "cli"),
    "{div} | {tdepth}"
  )
  expect_equal(
    nutpieR:::validate_chain_format("[{elapsed}] c{chain}", "text"),
    "[{elapsed}] c{chain}"
  )
  expect_equal(
    nutpieR:::validate_chain_format("[{elapsed}] {tdepth}", "text"),
    "[{elapsed}] {tdepth}"
  )
  expect_error(
    nutpieR:::validate_chain_format(c("{div}", "{grad}"), "cli"),
    "single non-missing string",
    fixed = TRUE
  )
  expect_error(
    nutpieR:::validate_chain_format(NA_character_, "cli"),
    "single non-missing string",
    fixed = TRUE
  )
  expect_error(
    nutpieR:::validate_chain_format("{chain}", "cli"),
    "unsupported token",
    fixed = TRUE
  )
  expect_error(
    nutpieR:::validate_chain_format("{spread}", "text"),
    "unsupported token",
    fixed = TRUE
  )
})

test_that("protected progress callbacks fail once without raising", {
  n <- 0L
  callback <- nutpieR:::protect_progress_callback(function(snapshot) {
    n <<- n + 1L
    stop("kaboom")
  })

  expect_silent(callback(list()))
  expect_equal(n, 1L)
  expect_silent(callback(list()))
  expect_equal(n, 1L)
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

  # divergent_draws are warmup-inclusive 0-based indices; with num_warmup = 10
  # the earliest (absolute 17) is sample-relative 1-indexed draw 17 - 10 + 1 = 8.
  summary <- nutpieR:::summarize_progress_snapshot(snapshot, num_warmup = 10L)

  expect_equal(summary$total_finished, 210)
  expect_equal(summary$total_draws, 400)
  expect_equal(summary$phase, "mixed")
  expect_equal(summary$total_divergences, 2L)
  expect_equal(summary$min_step_size, 0.015)
  expect_equal(summary$max_latest_num_steps, 63L)
  expect_equal(summary$first_divergence, "chain 2 draw 8")
  # status is "! div: 2 | <grad>" — divergence indicator present
  expect_match(summary$status, "div: 2", fixed = TRUE)
  # avg_num_steps = (900+1800)/(120+90) = 2700/210 ~= 12.86
  expect_match(summary$status, "grad/draw", fixed = TRUE)
})

test_that("div hint fires exactly once and only post-warmup", {
  hints <- nutpieR:::new_progress_hints()
  m1 <- testthat::capture_messages(nutpieR:::maybe_div_hint(hints, 3L))
  expect_match(m1, "div: divergent transitions detected", all = FALSE)

  # already warned -> no repeat
  m2 <- testthat::capture_messages(nutpieR:::maybe_div_hint(hints, 5L))
  expect_length(m2, 0L)

  # zero post-warmup divergences never fires
  hints2 <- nutpieR:::new_progress_hints()
  expect_length(
    testthat::capture_messages(nutpieR:::maybe_div_hint(hints2, 0L)),
    0L
  )
})

test_that("text callback emits the div hint once across calls", {
  div_snapshot <- list(
    list(chain = 1L, finished_draws = 50L, total_draws = 100L,
         divergences = 0L, tuning = FALSE, started = TRUE,
         latest_num_steps = 3L, total_num_steps = 150L,
         step_size = 0.5, runtime = 1.0, divergent_draws = integer()),
    list(chain = 2L, finished_draws = 50L, total_draws = 100L,
         divergences = 2L, tuning = FALSE, started = TRUE,
         latest_num_steps = 3L, total_num_steps = 150L,
         step_size = 0.5, runtime = 1.0, divergent_draws = c(40L, 45L))
  )
  cb <- nutpieR:::make_text_progress_callback(
    num_chains = 2L, num_warmup = 0L, num_draws = 100L,
    refresh = 1L
  )
  msgs <- testthat::capture_messages(cb(div_snapshot))
  expect_equal(sum(grepl("divergent transitions detected", msgs)), 1L)

  # advance draws so per-chain lines print again, but the hint must not repeat
  div_snapshot[[1]]$finished_draws <- 80L
  div_snapshot[[2]]$finished_draws <- 80L
  msgs2 <- testthat::capture_messages(cb(div_snapshot))
  expect_equal(sum(grepl("divergent transitions detected", msgs2)), 0L)
})

test_that("grad/draw baseline anchors at late warmup and pools exactly", {
  hints <- nutpieR:::new_progress_hints()
  num_warmup <- 100L  # late-warmup threshold = 0.75 * 100 = 75

  # Below threshold: no baseline yet, pooled average is NA.
  snap0 <- list(
    list(chain = 1L, finished_draws = 50L, total_num_steps = 5000),
    list(chain = 2L, finished_draws = 50L, total_num_steps = 5000)
  )
  nutpieR:::update_grad_baselines(hints, snap0, num_warmup)
  expect_length(hints$grad_baseline, 0L)
  expect_true(is.na(nutpieR:::pooled_grad_per_draw(hints, snap0)))

  # Cross threshold: each chain records (total, finished). No post-baseline
  # draws yet, so pooled average is still NA.
  snap1 <- list(
    list(chain = 1L, finished_draws = 80L, total_num_steps = 8000),
    list(chain = 2L, finished_draws = 80L, total_num_steps = 8000)
  )
  nutpieR:::update_grad_baselines(hints, snap1, num_warmup)
  expect_length(hints$grad_baseline, 2L)
  expect_true(is.na(nutpieR:::pooled_grad_per_draw(hints, snap1)))

  # +100 draws/chain costing +20000 steps/chain -> exactly 200 grad/draw pooled.
  snap2 <- list(
    list(chain = 1L, finished_draws = 180L, total_num_steps = 28000),
    list(chain = 2L, finished_draws = 180L, total_num_steps = 28000)
  )
  expect_equal(nutpieR:::pooled_grad_per_draw(hints, snap2), 200)

  # Baselines are sticky: a later update does not move them.
  nutpieR:::update_grad_baselines(hints, snap2, num_warmup)
  expect_equal(nutpieR:::pooled_grad_per_draw(hints, snap2), 200)
})

test_that("grad hint fires once above the threshold with rounded depth", {
  hints <- nutpieR:::new_progress_hints()
  # Below GRAD_HINT_THRESHOLD (128): nothing.
  expect_length(testthat::capture_messages(nutpieR:::maybe_grad_hint(hints, 50)), 0L)

  m <- testthat::capture_messages(nutpieR:::maybe_grad_hint(hints, 210))
  expect_match(m, "grad/draw: ~210 gradient evaluations per draw", all = FALSE)
  expect_match(m, "tree depth ~8", all = FALSE)  # round(log2(211)) = 8

  # Once only.
  expect_length(testthat::capture_messages(nutpieR:::maybe_grad_hint(hints, 300)), 0L)
})

test_that("short elapsed times render as less than one tenth second", {
  expect_equal(nutpieR:::format_progress_time(0), "0.0s")
  expect_equal(nutpieR:::format_progress_time(0.04), "<0.1s")
  expect_equal(nutpieR:::format_progress_time(0.1), "0.1s")
})

test_that("progress_messages_muffled detects active message handlers", {
  expect_false(nutpieR:::progress_messages_muffled())
  expect_true(suppressMessages(nutpieR:::progress_messages_muffled()))
  muffled <- FALSE
  msgs <- testthat::capture_messages(
    muffled <- nutpieR:::progress_messages_muffled()
  )
  expect_true(muffled)
  expect_equal(msgs, "")
})

test_that("cli grad hint waits for the late-warmup baseline", {
  updates <- list()
  fake_update <- function(set = NULL, status = NULL, extra = NULL, id = NULL, force = FALSE) {
    updates[[length(updates) + 1L]] <<- list(set = set, status = status, extra = extra)
  }
  callback <- nutpieR:::make_cli_progress_callback(
    num_chains = 1L, num_warmup = 100L, num_draws = 100L,
    id = "fake", update = fake_update, done = function(...) TRUE
  )

  early_high <- list(list(
    chain = 1L, finished_draws = 20L, total_draws = 200L,
    divergences = 0L, tuning = TRUE, started = TRUE,
    latest_num_steps = 200L, total_num_steps = 4000L,
    step_size = 0.5, runtime = 1, divergent_draws = integer()
  ))
  expect_equal(
    sum(grepl("grad/draw:", testthat::capture_messages(callback(early_high)))),
    0L
  )

  baseline <- early_high
  baseline[[1]]$finished_draws <- 80L
  baseline[[1]]$total_num_steps <- 16000L
  expect_equal(
    sum(grepl("grad/draw:", testthat::capture_messages(callback(baseline)))),
    0L
  )

  settled_high <- baseline
  settled_high[[1]]$finished_draws <- 100L
  settled_high[[1]]$total_num_steps <- 20000L
  expect_equal(
    sum(grepl("grad/draw:", testthat::capture_messages(callback(settled_high)))),
    1L
  )
})

test_that("end summary reports %-at-cap from per-draw diagnostics", {
  # depth column present -> cap is depth >= max_treedepth; 4 of 20 draws = 20%.
  diags <- list(
    chain = rep(1:2, each = 10),
    n_steps = rep(3, 20),
    depth = c(rep(10L, 3L), rep(2L, 7L), 10L, rep(2L, 9L)),
    step_size = rep(0.5, 20),
    diverging = rep(FALSE, 20)
  )
  expect_equal(nutpieR:::fraction_at_treedepth_cap(diags, 10L), 0.2)

  msgs <- testthat::capture_messages(
    nutpieR:::print_sampling_diagnostic_summary(
      diags, num_chains = 2L, elapsed = 1, max_treedepth = 10L
    )
  )
  joined <- paste(msgs, collapse = "")
  expect_match(joined, "20% of draws hit the max_treedepth cap", fixed = TRUE)

  # Below CAP_SUMMARY_THRESHOLD (5%): no advice line.
  diags_low <- diags
  diags_low$depth <- rep(2L, 20)
  msgs_low <- testthat::capture_messages(
    nutpieR:::print_sampling_diagnostic_summary(
      diags_low, num_chains = 2L, elapsed = 1, max_treedepth = 10L
    )
  )
  expect_false(grepl("max_treedepth cap", paste(msgs_low, collapse = "")))
})

test_that("end summary headline warns when treedepth cap is high without divergences", {
  diags <- list(
    chain = rep(1:2, each = 10),
    n_steps = rep(3, 20),
    depth = c(rep(10L, 4L), rep(2L, 16L)),
    step_size = rep(0.5, 20),
    diverging = rep(FALSE, 20)
  )

  msgs <- testthat::capture_messages(
    nutpieR:::print_sampling_diagnostic_summary(
      diags, num_chains = 2L, elapsed = 1, max_treedepth = 10L
    )
  )
  joined <- paste(msgs, collapse = "\n")
  expect_match(joined, "with no divergences, but 20% of draws hit the max_treedepth cap", fixed = TRUE)
  expect_false(grepl("with no divergences.", msgs[[1]], fixed = TRUE))
})

test_that("end summary escalates severe divergence fractions", {
  diags <- list(
    chain = rep(1:2, each = 10),
    n_steps = rep(3, 20),
    depth = rep(2L, 20),
    step_size = rep(0.5, 20),
    diverging = c(rep(TRUE, 3L), rep(FALSE, 17L))
  )

  msgs <- testthat::capture_messages(
    nutpieR:::print_sampling_diagnostic_summary(
      diags, num_chains = 2L, elapsed = 1, max_treedepth = 10L
    )
  )
  joined <- paste(msgs, collapse = "\n")
  expect_match(joined, "15.0% of post-warmup draws", fixed = TRUE)
  expect_match(joined, "results are not reliable", fixed = TRUE)
  expect_match(joined, "deeper problem, not just a tuning issue", fixed = TRUE)
  expect_false(grepl("Try increasing `target_accept`", joined, fixed = TRUE))
})

test_that("fraction_at_treedepth_cap falls back to n_steps when no depth column", {
  diags <- list(
    chain = rep(1L, 10),
    n_steps = c(rep(1023, 2), rep(7, 8))  # 2/10 >= 2^10 - 1 = 1023
  )
  expect_equal(nutpieR:::fraction_at_treedepth_cap(diags, 10L), 0.2)
  expect_true(is.na(nutpieR:::fraction_at_treedepth_cap(list(chain = 1L), 10L)))
})

test_that("fraction_at_treedepth_cap prefers the maxdepth_reached flag", {
  # maxdepth_reached is authoritative; depth/max_treedepth are ignored when present.
  diags <- list(
    maxdepth_reached = c(rep(TRUE, 3), rep(FALSE, 7)),
    depth = rep(2L, 10)  # would say 0% via the fallback
  )
  expect_equal(nutpieR:::fraction_at_treedepth_cap(diags, 10L), 0.3)
  # works even without max_treedepth, since the flag needs no threshold
  expect_equal(nutpieR:::fraction_at_treedepth_cap(diags, NULL), 0.3)
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

  result <- nutpieR:::format_status_tokens(snapshot, summary, "{div} | {grad} | {tdepth}")
  expect_match(result, "div:", fixed = TRUE)
  expect_match(result, "grad/draw", fixed = TRUE)
  expect_match(result, "tdepth:", fixed = TRUE)

  # div-only format
  div_only <- nutpieR:::format_status_tokens(snapshot, summary, "{div}")
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
  result <- nutpieR:::format_status_tokens(snapshot, summary, "{div} | {lag} | {grad}")
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
  expect_match(result, "408-465", fixed = TRUE)  # ASCII hyphen, text-safe

  # when all chains have same count, no range separator
  snapshot_equal <- list(
    list(finished_draws = 500L, total_draws = 1000L),
    list(finished_draws = 500L, total_draws = 1000L)
  )
  result_eq <- nutpieR:::format_chain_draw_range(snapshot_equal)
  expect_false(grepl("-", result_eq, fixed = TRUE))
  expect_match(result_eq, "1k", fixed = TRUE)
})

test_that("spread token: sticky, started-chains-only, percent range", {
  # Two chains 20% apart, median past the floor -> triggers.
  spread_snap <- list(
    list(chain = 1L, finished_draws = 30L, total_draws = 100L, started = TRUE),
    list(chain = 2L, finished_draws = 50L, total_draws = 100L, started = TRUE)
  )
  expect_true(nutpieR:::spread_triggered(spread_snap))
  expect_equal(nutpieR:::format_chain_spread(spread_snap, active = TRUE), "spread 30-50%")
  # Inactive -> empty (the callback latches activation).
  expect_equal(nutpieR:::format_chain_spread(spread_snap, active = FALSE), "")

  # Too close together -> no trigger.
  close_snap <- list(
    list(chain = 1L, finished_draws = 48L, total_draws = 100L, started = TRUE),
    list(chain = 2L, finished_draws = 52L, total_draws = 100L, started = TRUE)
  )
  expect_false(nutpieR:::spread_triggered(close_snap))

  # Below the median floor -> no trigger even with a wide spread.
  early_snap <- list(
    list(chain = 1L, finished_draws = 1L, total_draws = 100L, started = TRUE),
    list(chain = 2L, finished_draws = 18L, total_draws = 100L, started = TRUE)
  )
  expect_false(nutpieR:::spread_triggered(early_snap))

  # Unstarted (queued) chains are ignored, so they don't fake a spread.
  queued_snap <- list(
    list(chain = 1L, finished_draws = 60L, total_draws = 100L, started = TRUE),
    list(chain = 2L, finished_draws = 0L, total_draws = 100L, started = FALSE)
  )
  expect_false(nutpieR:::spread_triggered(queued_snap))
  expect_equal(nutpieR:::format_chain_spread(queued_snap, active = TRUE), "")

  # Finished chains are also ignored, so an 8-on-4 style queued second wave
  # does not look like sampler-induced spread.
  second_wave <- list(
    list(chain = 1L, finished_draws = 100L, total_draws = 100L, started = TRUE),
    list(chain = 2L, finished_draws = 100L, total_draws = 100L, started = TRUE),
    list(chain = 3L, finished_draws = 13L, total_draws = 100L, started = TRUE),
    list(chain = 4L, finished_draws = 5L, total_draws = 100L, started = TRUE)
  )
  expect_false(nutpieR:::spread_triggered(second_wave))
  expect_equal(nutpieR:::format_chain_spread(second_wave, active = TRUE), "spread 5-13%")
})

test_that("lag and spark ignore finished chains from queued earlier waves", {
  second_wave <- list(
    list(chain = 1L, finished_draws = 100L, total_draws = 100L, started = TRUE),
    list(chain = 2L, finished_draws = 100L, total_draws = 100L, started = TRUE),
    list(chain = 3L, finished_draws = 13L, total_draws = 100L, started = TRUE),
    list(chain = 4L, finished_draws = 13L, total_draws = 100L, started = TRUE)
  )

  expect_equal(nutpieR:::format_chain_lag(second_wave), "")
  expect_equal(nchar(nutpieR:::format_chain_spark(second_wave)), 2L)
  expect_true(grepl("▁▁", nutpieR:::format_chain_spark(second_wave), fixed = TRUE))
})

test_that("spread hint fires once via the cli callback", {
  trigger_snap <- list(
    list(chain = 1L, finished_draws = 30L, total_draws = 100L, divergences = 0L,
         tuning = FALSE, started = TRUE, latest_num_steps = 3L, total_num_steps = 90,
         step_size = 0.5, runtime = 1, divergent_draws = integer()),
    list(chain = 2L, finished_draws = 50L, total_draws = 100L, divergences = 0L,
         tuning = FALSE, started = TRUE, latest_num_steps = 3L, total_num_steps = 150,
         step_size = 0.5, runtime = 1, divergent_draws = integer())
  )
  cb <- nutpieR:::make_cli_progress_callback(
    num_chains = 2L, num_warmup = 0L, num_draws = 100L,
    id = "fake", update = function(...) TRUE, done = function(...) TRUE
  )
  msgs <- testthat::capture_messages(cb(trigger_snap))
  expect_equal(sum(grepl("chain progress is uneven", msgs)), 1L)
  # advance but stay spread; hint does not repeat
  trigger_snap[[1]]$finished_draws <- 60L
  trigger_snap[[2]]$finished_draws <- 90L
  msgs2 <- testthat::capture_messages(cb(trigger_snap))
  expect_equal(sum(grepl("chain progress is uneven", msgs2)), 0L)
})

test_that("format_chain_spark renders a glyph per chain, flat when together", {
  together <- list(
    list(chain = 1L, finished_draws = 100L, total_draws = 200L, started = TRUE),
    list(chain = 2L, finished_draws = 100L, total_draws = 200L, started = TRUE)
  )
  spark <- nutpieR:::format_chain_spark(together)
  expect_equal(nchar(spark), 2L)
  expect_true(grepl("▁", spark))  # both at baseline

  laggard <- list(
    list(chain = 1L, finished_draws = 180L, total_draws = 200L, started = TRUE),
    list(chain = 2L, finished_draws = 20L, total_draws = 200L, started = TRUE)
  )
  spark2 <- nutpieR:::format_chain_spark(laggard)
  expect_equal(nchar(spark2), 2L)
  # leader is flat, laggard is a taller glyph
  expect_false(identical(substr(spark2, 1, 1), substr(spark2, 2, 2)))
})

test_that("format_gradient_status accents above the absolute grad threshold", {
  # Accent fires at GRAD_HINT_THRESHOLD (128) leapfrog steps/draw, regardless of
  # max_treedepth. The glyph is ▲ (UTF-8) or ^ (fallback).
  result_warn <- nutpieR:::format_gradient_status(200.0)
  expect_match(result_warn, "[▲^]")
  expect_match(result_warn, "200.0 grad/draw", fixed = TRUE)

  # 100 is below the threshold -> no accent.
  result_ok <- nutpieR:::format_gradient_status(100.0)
  expect_false(grepl("[▲^]", result_ok))
  expect_match(result_ok, "100.0 grad/draw", fixed = TRUE)
})

test_that("make_text_progress_callback prints one line per chain at refresh interval", {
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
      divergences = 0L, tuning = TRUE, started = TRUE,
      latest_num_steps = 3L, total_num_steps = 300L,
      step_size = 0.5, runtime = 5.0,
      divergent_draws = integer()
    )
  )

  callback <- nutpieR:::make_text_progress_callback(
    num_chains = 2L, num_warmup = 400L, num_draws = 1000L,
    refresh = 50L
  )
  # Text-mode lines go to stderr via message(); each chain is one message.
  # No divergences here, so only the two per-chain lines appear (the div hint
  # has its own test).
  out <- capture_messages(callback(snapshot))

  # Should print 2 lines (one per chain), since finished=100 >= refresh=50
  expect_length(out, 2L)
  expect_match(out[1], "c1", fixed = TRUE)
  expect_match(out[2], "c2", fixed = TRUE)
  expect_match(out[1], "warmup", fixed = TRUE)
  expect_match(out[2], "div:", fixed = TRUE)

  # Second call with same snapshot should not print (since_last = 0 < 50)
  out2 <- capture_messages(callback(snapshot))
  expect_length(out2, 0L)
})

test_that("text progress reports phase-relative draws and forces final line", {
  callback <- nutpieR:::make_text_progress_callback(
    num_chains = 1L, num_warmup = 100L, num_draws = 80L,
    refresh = 70L
  )
  warmup <- list(list(
    chain = 1L, finished_draws = 70L, total_draws = 180L,
    divergences = 0L, tuning = TRUE, started = TRUE,
    latest_num_steps = 3L, total_num_steps = 210L,
    step_size = 0.5, runtime = 1, divergent_draws = integer()
  ))
  out1 <- capture_messages(callback(warmup))
  expect_match(out1, "warmup 70%  70/100", fixed = TRUE)

  sample <- warmup
  sample[[1]]$finished_draws <- 140L
  sample[[1]]$tuning <- FALSE
  sample[[1]]$total_num_steps <- 420L
  out2 <- capture_messages(callback(sample))
  expect_match(out2, "sample 50%  40/80", fixed = TRUE)

  done <- sample
  done[[1]]$finished_draws <- 180L
  done[[1]]$total_num_steps <- 540L
  out3 <- capture_messages(callback(done))
  expect_match(out3, "sample 100%  80/80", fixed = TRUE)
})

test_that("text progress supports the treedepth token", {
  callback <- nutpieR:::make_text_progress_callback(
    num_chains = 1L, num_warmup = 10L, num_draws = 10L,
    refresh = 1L,
    chain_format = "c{chain} {tdepth}"
  )
  snapshot <- list(list(
    chain = 1L, finished_draws = 5L, total_draws = 20L,
    divergences = 0L, tuning = TRUE, started = TRUE,
    latest_num_steps = 7L, total_num_steps = 35L,
    step_size = 0.5, runtime = 1, divergent_draws = integer()
  ))
  out <- testthat::capture_messages(callback(snapshot))
  expect_match(out, "c1 tdepth: 3", fixed = TRUE)
})

test_that("text grad token switches to per-chain late-warmup baseline", {
  callback <- nutpieR:::make_text_progress_callback(
    num_chains = 1L, num_warmup = 100L, num_draws = 100L,
    refresh = 1L
  )
  baseline <- list(list(
    chain = 1L, finished_draws = 80L, total_draws = 200L,
    divergences = 0L, tuning = TRUE, started = TRUE,
    latest_num_steps = 200L, total_num_steps = 16000L,
    step_size = 0.5, runtime = 1, divergent_draws = integer()
  ))
  invisible(capture_messages(callback(baseline)))

  settled <- baseline
  settled[[1]]$finished_draws <- 100L
  settled[[1]]$total_num_steps <- 16140L
  out <- capture_messages(callback(settled))
  expect_match(out, "7.0 grad/draw", fixed = TRUE)
  expect_false(grepl("161.4 grad/draw", paste(out, collapse = "\n"), fixed = TRUE))
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
  expect_equal(updates[[1]]$extra$phase, "warmup")  # all chains still tuning
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

test_that("progress arguments fail before sampling on malformed inputs", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  expect_error(
    nutpie_sample(
      test_models$bernoulli, data = bernoulli_data(),
      num_draws = 30, num_warmup = 30, num_chains = 1,
      seed = 1L, progress = "text", refresh = NA_integer_
    ),
    "refresh"
  )
  expect_error(
    nutpie_sample(
      test_models$bernoulli, data = bernoulli_data(),
      num_draws = 30, num_warmup = 30, num_chains = 1,
      seed = 1L, progress = "text", refresh = 1L,
      chain_format = c("{chain}", "{draws}")
    ),
    "chain_format"
  )
  expect_error(
    nutpie_sample(
      test_models$bernoulli, data = bernoulli_data(),
      num_draws = 30, num_warmup = 30, num_chains = 1,
      seed = 1L, progress = "cli", refresh = 1L,
      chain_format = "{chain}"
    ),
    "unsupported token",
    fixed = TRUE
  )
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

test_that("suppressMessages silences text progress callbacks", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  out <- capture.output(
    msgs <- testthat::capture_messages(
      suppressMessages(
        draws <- nutpie_sample(
          test_models$bernoulli, data = bernoulli_data(),
          num_draws = 30, num_warmup = 30, num_chains = 2,
          seed = 1L, refresh = 1L, progress = "text"
        )
      )
    ),
    type = "output"
  )
  expect_s3_class(draws, "draws_array")
  expect_equal(msgs, character())
  expect_false(grepl("sample 100%", paste(out, collapse = "\n"), fixed = TRUE))
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
  out <- capture.output(
    suppressMessages(
      draws <- nutpie_sample(
        test_models$bernoulli, data = bernoulli_data(),
        num_draws = 30, num_warmup = 30, num_chains = 1,
        seed = 1L, refresh = 1L, progress = "cli"
      )
    ),
    type = "output"
  )
  expect_s3_class(draws, "draws_array")
  expect_equal(call_count, 1L)
  expect_false(grepl("Error in", paste(out, collapse = "\n"), fixed = TRUE))
})
