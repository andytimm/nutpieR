# Tests for the GitHub #36 tbbmalloc_proxy safety machinery. The compile-time
# source patch and the dlsym probe are exercised in Rust / integration; here we
# cover the R-side progress gate, which is pure and injectable.

test_that("gate leaves progress untouched when the allocator is safe", {
  expect_identical(gate_progress_for_tbb("cli", safe = TRUE), "cli")
  expect_identical(gate_progress_for_tbb("text", safe = TRUE), "text")
  expect_identical(gate_progress_for_tbb("none", safe = TRUE), "none")
})

test_that("gate is a no-op for non-live modes even with an unsafe allocator", {
  # "none" never renders live progress, so there is nothing to gate.
  expect_identical(gate_progress_for_tbb("none", safe = FALSE), "none")
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
