# Regression coverage for GitHub #36: on macOS, Stan links the
# `tbbmalloc_proxy` allocator into model `.so`s by default, which crashes R
# during live progress reporting. nutpieR pins `TBB_LIBRARIES=tbb` on macOS
# to drop the proxy (matching Linux/Windows), unless the caller overrides it.

test_that("effective_compile_args drops the TBB malloc proxy on macOS", {
  expect_identical(
    nutpieR:::effective_compile_args(character(), sysname = "Darwin"),
    "TBB_LIBRARIES=tbb"
  )
  # Existing user args are preserved; the default is appended.
  expect_identical(
    nutpieR:::effective_compile_args("STANCFLAGS=foo", sysname = "Darwin"),
    c("STANCFLAGS=foo", "TBB_LIBRARIES=tbb")
  )
})

test_that("effective_compile_args is a no-op off macOS", {
  for (os in c("Linux", "Windows")) {
    expect_identical(
      nutpieR:::effective_compile_args(character(), sysname = os),
      character()
    )
    expect_identical(
      nutpieR:::effective_compile_args("STANCFLAGS=foo", sysname = os),
      "STANCFLAGS=foo"
    )
  }
})

test_that("a caller-supplied TBB_LIBRARIES overrides the macOS default", {
  # Opting back into the proxy must be respected, and must not be
  # double-appended.
  proxy <- "TBB_LIBRARIES=tbb tbbmalloc tbbmalloc_proxy"
  expect_identical(
    nutpieR:::effective_compile_args(proxy, sysname = "Darwin"),
    proxy
  )
  expect_identical(
    nutpieR:::effective_compile_args("TBB_LIBRARIES=tbb", sysname = "Darwin"),
    "TBB_LIBRARIES=tbb"
  )
})

test_that("the macOS default vs. proxy land in different cache slots", {
  # Folding the flag in above the cache layer is what invalidates stale
  # proxy-linked artifacts compiled by older nutpieR versions.
  v <- "TEST.0"
  bundle <- nutpieR:::inline_bundle("data {}")
  default <- nutpieR:::effective_compile_args(character(), sysname = "Darwin")
  proxy <- nutpieR:::effective_compile_args(
    "TBB_LIBRARIES=tbb tbbmalloc tbbmalloc_proxy", sysname = "Darwin")
  expect_false(
    nutpieR:::cache_key(bundle, v, character(), default) ==
      nutpieR:::cache_key(bundle, v, character(), proxy)
  )
})

test_that("compiled macOS models do not link tbbmalloc_proxy", {
  skip_on_cran()
  skip_if_not(Sys.info()[["sysname"]] == "Darwin", "macOS-only linkage check")
  skip_if(Sys.which("otool") == "", "otool not available")
  skip_if_not(
    identical(Sys.getenv("NUTPIER_RUN_SLOW_TESTS"), "1"),
    "slow: compiles a Stan model"
  )

  model <- nutpie_compile_model(
    code = "parameters { real x; } model { x ~ normal(0, 1); }",
    verbose = 0L
  )
  deps <- system2("otool", c("-L", shQuote(model$lib_path)), stdout = TRUE)
  tbb <- grep("tbb", deps, value = TRUE)
  expect_true(any(grepl("libtbb\\.dylib", tbb)))      # threading still linked
  expect_false(any(grepl("tbbmalloc_proxy", tbb)))    # proxy dropped (#36)
})
