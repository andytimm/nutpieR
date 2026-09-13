test_that("copyable density kernel sources explicitly export ABI definitions", {
  definitions <- c(
    "nutpier_density_kernel_abi_version",
    "nutpier_density_kernel_bind",
    "nutpier_density_kernel_destroy",
    "nutpier_density_kernel_workspace",
    "nutpier_density_kernel_workspace_destroy",
    "nutpier_density_kernel_evaluate"
  )
  sources <- c(
    system.file("examples", "density-kernel", "gaussian.c", package = "nutpieR"),
    test_path("density-kernel-v1", "gaussian.c")
  )
  for (source in sources) {
    lines <- readLines(source)
    for (definition in definitions) {
      expect_equal(length(grep(paste0(
        "^NUTPIER_DENSITY_KERNEL_EXPORT (uint32_t|int32_t|void) ",
        definition, "\\("), lines)), 1L, info = source)
    }
  }
})

test_that("custom density kernels preserve sampling, transforms and output behavior", {
  skip_if(Sys.getenv("NUTPIER_RUN_DENSITY_KERNEL_TESTS") != "1",
          "set NUTPIER_RUN_DENSITY_KERNEL_TESTS=1 after a release install")
  script <- normalizePath(test_path("density-kernel-examples", "sample-examples.R"), mustWork = TRUE)
  fixtures <- normalizePath(test_path("density-kernel-v1"), mustWork = TRUE)
  log <- tempfile("density-kernel-examples-", fileext = ".log")
  status <- system2(file.path(R.home("bin"), "Rscript"),
                    c(shQuote(script), shQuote(fixtures)),
                    stdout = log, stderr = log, timeout = 240)
  output <- paste(readLines(log, warn = FALSE), collapse = "\n")
  expect_equal(status, 0L, info = paste(log, output))
  expect_match(output, "BROAD SAMPLING PASSED")
})
