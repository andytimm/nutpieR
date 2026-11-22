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
  supports <- function(n, height) {
    nutpieR:::supports_multiline_progress(
      n, cli_version = package_version("3.6.6.9000"), ansi = TRUE,
      multiline = TRUE, unsunk = TRUE, height = height
    )
  }
  expect_false(supports(4L, 5L))
  expect_true(supports(4L, 6L))
  expect_true(supports(12L, 24L))
  expect_true(supports(4L, NA_integer_))
  expect_false(supports(5L, NA_integer_))
})
