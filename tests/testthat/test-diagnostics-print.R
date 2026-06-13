# Failure-flag behaviour of print.nutpie_diagnostics(). All cases are synthetic
# (no compiled model needed): we build a posterior::draws_array, attach the same
# attributes nutpie_sample() does (sample.R:306-311), and assert on the printed
# output. cat() lines go to stdout (expect_output); cli_alert_* flag lines go to
# the message stream (capture_messages).

# Build a draws_array with mu chain-means `mu_means` (one per chain) plus a
# well-mixed sigma, and attach a diagnostics attribute of the shape
# nutpie_diagnostics() expects.
make_diag_object <- function(mu_means, energy_pattern = "healthy",
                             diverging = NULL, n_iter = 200L) {
  set.seed(42)
  n_chain <- length(mu_means)
  arr <- array(0, dim = c(n_iter, n_chain, 2L),
               dimnames = list(NULL, NULL, c("mu", "sigma")))
  for (ch in seq_len(n_chain)) {
    arr[, ch, "mu"] <- stats::rnorm(n_iter, mu_means[ch], 0.1)
    arr[, ch, "sigma"] <- stats::rnorm(n_iter, 1, 0.1)
  }
  draws <- posterior::as_draws_array(arr)

  n <- n_iter * n_chain
  energy <- if (identical(energy_pattern, "healthy")) {
    rep(c(0, 10), length.out = n)            # high E-BFMI everywhere
  } else {
    rep(seq_len(n_iter), times = n_chain)    # slow drift -> low E-BFMI
  }
  if (is.null(diverging)) diverging <- rep(FALSE, n)

  attr(draws, "diagnostics") <- list(
    chain            = rep(seq_len(n_chain), each = n_iter),
    draw             = rep(seq_len(n_iter), times = n_chain),
    diverging        = diverging,
    depth            = rep(2L, n),
    step_size_bar    = rep(0.5, n),
    mean_tree_accept = rep(0.9, n),
    energy           = energy
  )
  attr(draws, "num_chains") <- n_chain
  nutpie_diagnostics(draws)
}

test_that("print always shows Max R-hat / ESS / E-BFMI info lines", {
  diag <- make_diag_object(c(0, 0, 0, 0))
  out <- paste(capture.output(suppressMessages(print(diag))), collapse = "\n")
  expect_match(out, "Max R-hat")
  expect_match(out, "Min Bulk-ESS")
  expect_match(out, "Min Tail-ESS")
  expect_match(out, "Min E-BFMI")
  expect_match(out, "posterior::summarize_draws(draws)", fixed = TRUE)
  # Developer field dump is gone.
  expect_false(grepl("Available fields", out, fixed = TRUE))
})

test_that("a non-mixing (bimodal) fit flags R-hat/ESS and names the parameter", {
  # Chains split across mu = -2 / +2: clean geometry, but R-hat blows out and
  # ESS collapses for mu. Short chains (n_iter = 60) drop both bulk and tail ESS
  # below 400 so the folded "Bulk/Tail-ESS" clause is exercised.
  diag <- make_diag_object(c(-2, -2, 2, 2), n_iter = 60L)
  msgs <- paste(testthat::capture_messages(
    suppressWarnings(print(diag))
  ), collapse = "")
  expect_match(msgs, "chains may not have mixed", fixed = TRUE)
  expect_match(msgs, "`mu`", fixed = TRUE)
  # Bulk + tail trip on the same parameter -> one folded clause, not two.
  expect_match(msgs, "Bulk/Tail-ESS", fixed = TRUE)
  # The summarize_draws pointer is NOT repeated in the flag; it lives once in
  # the footer (stdout), not the message stream.
  expect_false(grepl("summarize_draws", msgs, fixed = TRUE))
})

test_that("a clean fit emits no failure flags", {
  diag <- make_diag_object(c(0, 0, 0, 0))
  msgs <- paste(testthat::capture_messages(print(diag)), collapse = "")
  expect_false(grepl("may not have mixed", msgs, fixed = TRUE))
  expect_false(grepl("E-BFMI below", msgs, fixed = TRUE))
  expect_false(grepl("divergent transition", msgs, fixed = TRUE))
})

test_that("low-E-BFMI fit flags energy independently of clean geometry", {
  diag <- make_diag_object(c(0, 0, 0, 0), energy_pattern = "drift")
  msgs <- paste(testthat::capture_messages(print(diag)), collapse = "")
  expect_match(msgs, "chains had an E-BFMI below 0.3", fixed = TRUE)
})

test_that("single-chain print skips R-hat with a note but still shows ESS", {
  diag <- make_diag_object(c(0))
  out <- paste(capture.output(suppressMessages(print(diag))), collapse = "\n")
  expect_match(out, "needs >= 2 chains", fixed = TRUE)
  expect_match(out, "Min Bulk-ESS")
})

test_that("divergence flag uses the severity-gated vocabulary", {
  # 15% of draws diverge -> severe escalation wording.
  n_iter <- 200L
  n_chain <- 2L
  n <- n_iter * n_chain
  diverging <- rep(FALSE, n)
  diverging[seq_len(round(0.15 * n))] <- TRUE
  diag <- make_diag_object(c(0, 0), diverging = diverging)
  msgs <- paste(testthat::capture_messages(print(diag)), collapse = "")
  expect_match(msgs, "results are not reliable", fixed = TRUE)
  expect_match(msgs, "deeper problem, not just a tuning issue", fixed = TRUE)
})

test_that("worst_rhat_ess returns extrema and offending variables", {
  arr <- array(0, dim = c(200L, 4L, 2L),
               dimnames = list(NULL, NULL, c("mu", "sigma")))
  set.seed(1)
  means <- c(-2, -2, 2, 2)
  for (ch in seq_len(4L)) {
    arr[, ch, "mu"] <- stats::rnorm(200, means[ch], 0.1)
    arr[, ch, "sigma"] <- stats::rnorm(200, 1, 0.1)
  }
  draws <- posterior::as_draws_array(arr)
  res <- nutpieR:::worst_rhat_ess(draws)
  expect_true(res$max_rhat > 1.01)
  expect_identical(res$max_rhat_var, "mu")
  expect_true(res$min_ess_bulk < 400)
  expect_identical(res$min_ess_bulk_var, "mu")

  # Single chain: R-hat unavailable, ESS still computed.
  single <- posterior::as_draws_array(arr[, 1, , drop = FALSE])
  res1 <- nutpieR:::worst_rhat_ess(single)
  expect_null(res1$max_rhat)
  expect_false(is.null(res1$min_ess_bulk))
})
