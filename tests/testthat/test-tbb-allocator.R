# Tests for the GitHub #36 tbbmalloc_proxy safety machinery. The compile-time
# source patch and the dlsym probe are exercised in Rust / integration; here we
# cover the R-side progress gate, which is pure and injectable.

test_that("gate leaves progress untouched when the allocator is safe", {
  expect_identical(gate_progress_for_tbb("cli", safe = TRUE), "cli")
  expect_identical(gate_progress_for_tbb("text", safe = TRUE), "text")
  expect_identical(gate_progress_for_tbb("none", safe = TRUE), "none")
})

test_that("gate warns for progress = 'none' too but leaves the mode alone", {
  # The session is at risk for every allocation, not just live progress, so an
  # unsafe allocator warns even in "none" mode — but the mode stays "none".
  withr::local_options(nutpieR.tbb_gate_warned = NULL)
  expect_warning(
    expect_identical(gate_progress_for_tbb("none", safe = FALSE), "none"),
    "tbbmalloc_proxy"
  )
})

test_that("gate downgrades live progress and warns once when unsafe", {
  withr::local_options(nutpieR.tbb_gate_warned = NULL)

  expect_warning(
    expect_identical(gate_progress_for_tbb("cli", safe = FALSE), "none"),
    "tbbmalloc_proxy"
  )
  # Second call in the same session downgrades silently (warned-once flag set).
  expect_no_warning(
    expect_identical(gate_progress_for_tbb("text", safe = FALSE), "none")
  )
  expect_true(getOption("nutpieR.tbb_gate_warned"))
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
