test_that("kernel sampling discards failed runs and keeps the binding usable", {
  skip_if(Sys.getenv("NUTPIER_RUN_BYOK_TESTS") != "1",
          "set NUTPIER_RUN_BYOK_TESTS=1 after a release install")
  fixtures <- normalizePath(test_path("byok-v1"), mustWork = TRUE)
  script <- normalizePath(test_path("byok-core", "sample-case.R"), mustWork = TRUE)
  log <- tempfile("byok-sampling-", fileext = ".log")
  status <- system2(file.path(R.home("bin"), "Rscript"),
    c(shQuote(script), shQuote(fixtures)), stdout = log, stderr = log, timeout = 180)
  output <- paste(readLines(log, warn = FALSE), collapse = "\n")
  expect_equal(status, 0L, info = paste("log", log, output))
  expect_match(output, "BYOK SAMPLING REGRESSIONS PASSED")
})
