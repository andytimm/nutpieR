test_that("native kernels preserve sampling, transforms and output behavior", {
  skip_if(Sys.getenv("NUTPIER_RUN_BYOK_TESTS") != "1",
          "set NUTPIER_RUN_BYOK_TESTS=1 after a release install")
  script <- normalizePath(test_path("byok-examples", "sample-examples.R"), mustWork = TRUE)
  fixtures <- normalizePath(test_path("byok-v1"), mustWork = TRUE)
  log <- tempfile("byok-examples-", fileext = ".log")
  status <- system2(file.path(R.home("bin"), "Rscript"),
                    c(shQuote(script), shQuote(fixtures)),
                    stdout = log, stderr = log, timeout = 240)
  output <- paste(readLines(log, warn = FALSE), collapse = "\n")
  expect_equal(status, 0L, info = paste(log, output))
  expect_match(output, "BROAD SAMPLING PASSED")
})
