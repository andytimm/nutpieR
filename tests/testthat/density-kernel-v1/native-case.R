# Standalone, bounded-child integration case. Never source in the test runner.
args <- commandArgs(trailingOnly = TRUE)
fixture_dir <- normalizePath(args[[1]], mustWork = TRUE)
mode <- as.integer(args[[2]])
library(nutpieR)
work <- tempfile("density-kernel-fixture-")
dir.create(work)
file.copy(file.path(fixture_dir, "gaussian.c"), work)
old <- setwd(work)
include <- system.file("include", package = "nutpieR")
stopifnot(nzchar(include))
Sys.setenv(PKG_CPPFLAGS = paste0('-I"', include, '" -DFIXTURE_MODE=', mode))
status <- system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", "gaussian.c"),
                  stdout = "build.log", stderr = "build.log", timeout = 90)
if (status != 0L) stop(paste(readLines("build.log"), collapse = "\n"))
lib <- normalizePath(paste0("gaussian", .Platform$dynlib.ext))
ref <- nutpie_compile_model(file.path(fixture_dir, if (mode >= 15L) "positive.stan" else "gaussian.stan"))
data <- list(n = 2L, mu = 1, sigma = 2)
error <- function(expr, pattern) {
  e <- tryCatch({ force(expr); NULL }, error = identity)
  stopifnot(inherits(e, "error"), grepl(pattern, conditionMessage(e), ignore.case = TRUE))
}
if (mode %in% c(12L, 13L)) {
  error(nutpie_attach_density_kernel(ref, lib, data), if (mode == 12L) "bind failure" else "version|ABI")
} else {
  bound <- nutpie_attach_density_kernel(ref, lib, data)
  stopifnot(identical(class(bound), c("nutpie_density_kernel_model", "nutpie_model")))
  stopifnot(identical(ls(bound), sort(c("reference", "data_json", "bs_ptr",
    "kernel_ptr", "kernel_path", "lib_path", "unc_names", "ndim"))))
  points <- rbind(c(-1, 2), c(3, -2), c(0.5, 1))
  if (mode == 11L) {
    error(nutpie_validate_density_kernel(bound, points), "workspace failure")
  } else {
    report <- nutpie_validate_density_kernel(bound, points)
    stopifnot(identical(report$status, if (mode %in% c(0:2, 15L)) "pass" else "fail"))
    if (mode == 3L) stopifnot(report$constant_offset$status == "possible_constant_offset")
    if (mode == 14L) stopifnot(report$repeatability$status == "fail")
  }
  if (mode %in% 0:2) {
    # Same values, different formatting/order/number notation for fixed producer.
    equal <- nutpie_attach_density_kernel(ref, lib, '{ "sigma": 2e0, "mu": 1.0, "n": 2 }')
    stopifnot(nutpie_validate_density_kernel(equal, points)$status == "pass")
    if (mode == 1L) error(nutpie_attach_density_kernel(ref, lib, list(n=2L, mu=2, sigma=2)), "fixed data")
    if (mode != 1L) {
      changed <- nutpie_attach_density_kernel(ref, lib, list(n=3L, mu=-2, sigma=0.5))
      stopifnot(nutpie_validate_density_kernel(changed, rbind(c(1,2,3), c(-1,0,2)))$status == "pass")
      stopifnot(nutpie_validate_density_kernel(bound, points)$status == "pass")
    }
    # Bypass attach's matching data snapshot only to probe bind guardrails.
    error(nutpieR:::density_kernel_bind(bound$bs_ptr, lib, '{"n":3,"mu":1,"sigma":2}'), "dimension")
    wrong <- nutpie_compile_model(file.path(fixture_dir, "layout.stan"))
    error(nutpie_attach_density_kernel(wrong, lib, data),
          'layout mismatch at coordinate 1.*expected "x.1".*received "y.1"')
    saveRDS(bound, "dead.rds")
    dead <- readRDS("dead.rds")
    error(nutpie_validate_density_kernel(dead, points), "dead|pointer|reattach|rebind")
    rm(equal, dead); gc()
    stopifnot(nutpie_validate_density_kernel(bound, points)$status == "pass")
  }
}
cat("density kernel fixture mode", mode, "passed\n")
setwd(old)
unlink(work, recursive = TRUE)
