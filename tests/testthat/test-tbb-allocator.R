# Tests for the GitHub #36 tbbmalloc_proxy safety machinery. The compile-time
# source patch and the dlsym probe are exercised in Rust / integration; here we
# cover the R-side progress gate, which is pure and injectable.

test_that("gate leaves progress untouched when the allocator is safe", {
  expect_identical(gate_progress_for_tbb("cli", safe = TRUE), "cli")
  expect_identical(gate_progress_for_tbb("text", safe = TRUE), "text")
  expect_identical(gate_progress_for_tbb("none", safe = TRUE), "none")
})

test_that("gate stops sampling when an unsafe allocator is loaded", {
  for (mode in c("none", "cli", "text")) {
    expect_error(
      gate_progress_for_tbb(mode, safe = FALSE),
      "Restart R"
    )
  }
})

test_that("sampling checks allocator safety before opening a model", {
  testthat::local_mocked_bindings(
    tbb_proxy_live_progress_safe = function() FALSE,
    bs_open = function(...) stop("model was opened"),
    .package = "nutpieR"
  )
  expect_error(
    nutpie_sample("not-used", num_draws = 1, num_chains = 1, refresh = 0),
    "Restart R"
  )
})

test_that("sampling rechecks allocator safety immediately after opening", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  calls <- 0L
  testthat::local_mocked_bindings(
    tbb_proxy_live_progress_safe = function() {
      calls <<- calls + 1L
      calls == 1L
    },
    .package = "nutpieR"
  )
  expect_error(
    nutpie_sample(
      test_models$bernoulli, data = bernoulli_data(),
      num_draws = 1, num_chains = 1, refresh = 0
    ),
    "Restart R"
  )
  expect_equal(calls, 2L)
})

test_that("live-progress safety probe is callable and TRUE with no proxy loaded", {
  # On a fresh session with no compiled model loaded (and on all non-macOS
  # platforms) no tbbmalloc_proxy is in the process, so the probe is TRUE.
  # If an earlier test in the run compiled a model, the patched proxy is loaded
  # and this is still TRUE — the only FALSE case is a stale unpatched proxy.
  expect_true(tbb_proxy_live_progress_safe())
})

test_that("bundled TBB proxy header still matches the verbatim splice (#36)", {
  # A BridgeStan/TBB bump that changed impl_malloc_usable_size would make the
  # exact-string splice in ensure_safe_tbb_proxy() silently decline, reverting
  # macOS users to the unpatched proxy. Assert the bundled header still contains
  # either the stock function text or nutpieR's marker, so such a bump fails
  # loudly in CI/dev.
  skip_on_os(c("windows", "linux", "solaris"))
  strings <- tbb_patch_strings()
  skip_if(length(strings) < 2L, "TBB patch strings unavailable (non-macOS build).")
  stock <- strings[[1L]]
  marker <- strings[[2L]]

  bs_root <- file.path(path.expand("~"), ".bridgestan")
  skip_if_not(dir.exists(bs_root), "BridgeStan sources not downloaded yet.")
  headers <- Sys.glob(file.path(
    bs_root, "bridgestan-*", "stan", "lib", "stan_math", "lib",
    "tbb_*", "src", "tbbmalloc", "proxy_overload_osx.h"
  ))
  skip_if(length(headers) == 0L, "No bundled TBB proxy header found.")

  for (h in headers) {
    content <- paste(readLines(h, warn = FALSE), collapse = "\n")
    expect_true(
      grepl(stock, content, fixed = TRUE) || grepl(marker, content, fixed = TRUE),
      info = paste0(
        "Bundled TBB proxy header neither matches the stock function text nor ",
        "carries the nutpieR marker; the verbatim splice in ",
        "ensure_safe_tbb_proxy() has silently stopped applying: ", h
      )
    )
  }
})
