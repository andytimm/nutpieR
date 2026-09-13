# Run with source(system.file("examples/density-kernel/run.R", package = "nutpieR"))
# Requires the C/C++ toolchain used for R packages and Stan compilation.
local({
  library(nutpieR)
  example_dir <- system.file("examples/density-kernel", package = "nutpieR", mustWork = TRUE)
  work <- tempfile("nutpier-density-kernel-example-")
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
  data <- list(n = 2L, mu = 1, sigma = 2)
  layout <- nutpie_density_layout(reference, data)
  stopifnot(identical(layout$names, c("x.1", "x.2")),
            identical(layout$layout, "x.1\nx.2"))
  bound <- nutpie_attach_density_kernel(reference,
    normalizePath(paste0("gaussian", .Platform$dynlib.ext)), data = data)
  # Default: 10 broad random points. Reference mode adds a short pilot:
  # one BridgeStan-only chain, 200 warmup + 10 draws, then four random points.
  report <- nutpie_validate_density_kernel(bound, seed = 42, method = "reference")
  print(report)
  stopifnot(report$status == "pass")
  draws <- nutpie_sample(bound, num_draws = 500, num_warmup = 300,
                         num_chains = 2, seed = 42, progress = "none")
  print(posterior::summarise_draws(draws))
  invisible(list(layout = layout, bound = bound, report = report, draws = draws))
})
