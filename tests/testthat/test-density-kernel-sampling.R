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

# Each mode is isolated: a hang, longjmp, or late destructor cannot corrupt the
# test runner. Interrupt cases use a separate bounded helper, never fork R.
for (case in c("cleanup", "callback", "callback-interrupt", "interrupt-none", "interrupt-text", "interrupt-callback")) {
  local({
    mode <- case
    test_that(paste("kernel lifecycle joins and permits reuse:", mode), {
      skip_if(Sys.getenv("NUTPIER_RUN_DENSITY_KERNEL_TESTS") != "1",
              "set NUTPIER_RUN_DENSITY_KERNEL_TESTS=1 after a release install")
      if (startsWith(mode, "interrupt-"))
        skip_on_os("windows") # Unix SIGINT is not portable to Windows Rscript.
      fixtures <- normalizePath(test_path("density-kernel-v1"), mustWork=TRUE)
      core <- normalizePath(test_path("density-kernel-core"), mustWork=TRUE)
      log <- tempfile(paste0("density-kernel-", mode, "-"), fileext=".log")
      status <- system2(file.path(R.home("bin"), "Rscript"),
        c(shQuote(file.path(core, "lifecycle-case.R")), shQuote(fixtures),
          shQuote(core), shQuote(mode)), stdout=log, stderr=log, timeout=180)
      output <- paste(readLines(log, warn=FALSE), collapse="\n")
      expect_equal(status, 0L, info=paste("log", log, output))
      expect_match(output, paste("density kernel lifecycle", mode, "passed"), fixed=TRUE)
    })
  })
}
