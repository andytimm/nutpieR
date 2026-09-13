test_that("kernel sampling discards failed runs and keeps the binding usable", {
  skip_if(Sys.getenv("NUTPIER_RUN_DENSITY_KERNEL_TESTS") != "1",
          "set NUTPIER_RUN_DENSITY_KERNEL_TESTS=1 after a release install")
  fixtures <- normalizePath(test_path("density-kernel-v1"), mustWork = TRUE)
  script <- normalizePath(test_path("density-kernel-core", "sample-case.R"), mustWork = TRUE)
  log <- tempfile("density-kernel-sampling-", fileext = ".log")
  status <- system2(file.path(R.home("bin"), "Rscript"),
    c(shQuote(script), shQuote(fixtures)), stdout = log, stderr = log, timeout = 180)
  output <- paste(readLines(log, warn = FALSE), collapse = "\n")
  expect_equal(status, 0L, info = paste("log", log, output))
  expect_match(output, "density kernel sampling regressions passed")
})
