# Run with source(system.file("examples/byok/run.R", package = "nutpieR"))
# Requires the C/C++ toolchain used for R packages and Stan compilation.
local({
  library(nutpieR)
  example_dir <- system.file("examples/byok", package = "nutpieR", mustWork = TRUE)
  work <- tempfile("nutpier-byok-example-")
  dir.create(work)
  file.copy(file.path(example_dir, c("gaussian.c", "gaussian.stan")), work)
  old <- setwd(work)
  on.exit(setwd(old))
  old_flags <- Sys.getenv("PKG_CPPFLAGS", unset = NA_character_)
  on.exit(if (is.na(old_flags)) Sys.unsetenv("PKG_CPPFLAGS") else
    Sys.setenv(PKG_CPPFLAGS = old_flags), add = TRUE)
  include <- system.file("include", package = "nutpieR", mustWork = TRUE)
  Sys.setenv(PKG_CPPFLAGS = paste0('-I"', include, '"'))
  status <- system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", "gaussian.c"))
  if (status != 0L) stop("Could not compile the example kernel.")

  reference <- nutpie_compile_model("gaussian.stan")
  bound <- nutpie_attach_kernel(reference,
    normalizePath(paste0("gaussian", .Platform$dynlib.ext)),
    data = list(n = 2L, mu = 1, sigma = 2))
  # Default: 10 broad random points. Reference mode adds a short pilot:
  # one BridgeStan-only chain, 200 warmup + 10 draws, then four random points.
  report <- nutpie_validate_kernel(bound, seed = 42, method = "reference")
  print(report)
  stopifnot(report$status == "pass")
  draws <- nutpie_sample(bound, num_draws = 500, num_warmup = 300,
                         num_chains = 2, seed = 42, progress = "none")
  print(posterior::summarise_draws(draws))
  invisible(list(bound = bound, report = report, draws = draws))
})
