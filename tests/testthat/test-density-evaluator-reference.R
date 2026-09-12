test_that("bound density evaluators retain their reference while sampling RNG changes", {
  skip_if(Sys.getenv("NUTPIER_RUN_BYOLD_TESTS") != "1",
          "set NUTPIER_RUN_BYOLD_TESTS=1 after a release install")
  script <- normalizePath(test_path("byold-core", "reference-case.R"), mustWork = TRUE)
  log <- tempfile("byold-reference-", fileext = ".log")
  status <- system2(file.path(R.home("bin"), "Rscript"), shQuote(script),
                    stdout = log, stderr = log, timeout = 180)
  output <- paste(readLines(log, warn = FALSE), collapse = "\n")
  expect_equal(status, 0L, info = paste(log, output))
  expect_match(output, "BOUND REFERENCE REGRESSIONS PASSED")
})
