test_that("height detection prefers measured tty rows and handles missing panes", {
  measured <- function() c(width = 80L, height = 7L)
  no_tty <- function() stop("not a tty")
  height <- nutpieR:::progress_console_height
  expect_equal(height(measured, "24"), 7L)
  expect_equal(height(no_tty, "24"), 24L)
  expect_equal(height(function() c(height = 0L), "9"), 9L)
  for (invalid in c("", "0", "-2", "Inf", "12 rows")) {
    expect_identical(height(no_tty, invalid), NA_integer_)
  }
})

test_that("per-chain rows require space for a header and cursor", {
  supports <- nutpieR:::progress_chain_rows_fit
  expect_false(supports(4L, 5L))
  expect_true(supports(4L, 6L))
  expect_true(supports(12L, 24L))
  expect_true(supports(4L, NA_integer_))
  expect_false(supports(5L, NA_integer_))
})

test_that("chain rows grow from a short pane without flapping near the threshold", {
  testthat::local_mocked_bindings(progress_messages_muffled = function() FALSE,
                                 .package = "nutpieR")
  rows <- 3L
  ids <- c("header", "one", "two")
  updated <- cleared <- character()
  callback <- nutpieR:::make_multi_cli_progress_callback(
    2L, 100L, 100L, ids = ids, height = function() rows,
    update = function(id, ...) updated <<- c(updated, id),
    done = function(id, result) cleared <<- c(cleared, id)
  )
  snapshot <- lapply(1:2, function(i) list(chain = i, finished_draws = 0L))
  tick <- function(height) {
    rows <<- height
    updated <<- character()
    callback(snapshot)
    updated
  }
  expect_identical(tick(3L), "header")
  expect_length(tick(NA_integer_), 0L)
  expect_length(tick(4L), 0L)
  expect_identical(tick(5L), ids)
  expect_length(tick(NA_integer_), 0L)
  expect_length(tick(4L), 0L)
  expect_length(cleared, 0L)
  expect_identical(tick(3L), "header")
  expect_setequal(cleared, ids[-1L])
})
