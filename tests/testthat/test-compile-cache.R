# Tests for the nutpie_compile_model() compile cache.
#
# Each compile costs ~20s, so we group axes per test rather than spreading
# them across one-claim-per-block. The inline-cache test isolates
# R_USER_CACHE_DIR so its nutpie_clear_cache() call doesn't wipe entries
# helper-models.R seeded for the rest of the suite.

skip_if_no_make <- function() {
  if (Sys.which("make") == "") testthat::skip("`make` not on PATH")
}

make_temp_stan <- function(content = "parameters { real x; } model { x ~ normal(0, 1); }") {
  d <- tempfile("nutpieR-cache-test-")
  dir.create(d, recursive = TRUE)
  f <- file.path(d, "model.stan")
  writeLines(content, f)
  f
}

test_that("stan_file cache hits, edits invalidate, and cache = FALSE bypasses", {
  skip_if_no_make()

  stan <- make_temp_stan()
  on.exit(unlink(dirname(stan), recursive = TRUE), add = TRUE)

  # Cold compile + warm hit: same path, mtime unchanged, warm call returns
  # in <2s (mtime check, no make invocation).
  m1 <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_true(file.exists(m1$lib_path))
  art_mtime <- file.info(m1$lib_path)$mtime

  warm <- system.time(
    m2 <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  )[["elapsed"]]
  expect_equal(m2$lib_path, m1$lib_path)
  expect_equal(file.info(m2$lib_path)$mtime, art_mtime)
  expect_lt(warm, 2)

  # Edit the .stan and bump mtime past the artifact -- recompile must fire.
  Sys.sleep(1.1)
  writeLines("parameters { real x; } model { x ~ normal(0, 2); }", stan)
  Sys.setFileTime(stan, Sys.time())
  m3 <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_true(file.info(m3$lib_path)$mtime > art_mtime)

  # cache = FALSE: artifact must land in a tempdir, not the in-place slot.
  m_cold <- nutpie_compile_model(stan_file = stan, verbose = 0L, cache = FALSE)
  expect_false(m_cold$lib_path == m3$lib_path)
  expect_true(startsWith(
    normalizePath(m_cold$lib_path, mustWork = TRUE),
    normalizePath(tempdir(), mustWork = TRUE)
  ))
})

test_that("inline cache: hit by content, separate entries per content, clear wipes", {
  skip_if_no_make()

  td <- tempfile("nutpieR-test-cache-")
  dir.create(td, recursive = TRUE)
  withr::local_envvar(c(R_USER_CACHE_DIR = td))
  withr::defer(unlink(td, recursive = TRUE))

  src_a <- "parameters { real a; } model { a ~ normal(0, 1); }"
  src_b <- "parameters { real b; } model { b ~ normal(0, 2); }"

  m1 <- nutpie_compile_model(code = src_a, verbose = 0L)
  expect_true(startsWith(
    normalizePath(m1$lib_path, mustWork = TRUE),
    normalizePath(nutpie_cache_dir(), mustWork = TRUE)
  ))

  warm <- system.time(
    m2 <- nutpie_compile_model(code = src_a, verbose = 0L)
  )[["elapsed"]]
  expect_equal(m2$lib_path, m1$lib_path)
  expect_lt(warm, 2)

  m3 <- nutpie_compile_model(code = src_b, verbose = 0L)
  expect_false(m3$lib_path == m1$lib_path)

  cache_root <- tools::R_user_dir("nutpieR", "cache")
  expect_true(dir.exists(cache_root))
  nutpie_clear_cache()
  expect_false(dir.exists(cache_root))
})

test_that("inline cache key folds in content, BridgeStan version, and sorted flags", {
  # Pure key derivation -- no compile cost. Verifies the four invariants
  # that make the cache safe to share across runs.
  v <- nutpieR:::bridgestan_version()
  k_base <- nutpieR:::inline_cache_key("data {}", v, character(), character())

  # Content matters.
  expect_false(k_base == nutpieR:::inline_cache_key(
    "data { int N; }", v, character(), character()))
  # stanc_args matter.
  expect_false(k_base == nutpieR:::inline_cache_key(
    "data {}", v, "--O1", character()))
  # compile_args matter.
  expect_false(k_base == nutpieR:::inline_cache_key(
    "data {}", v, character(), "STANCFLAGS=foo"))
  # BridgeStan version matters.
  expect_false(nutpieR:::inline_cache_key("data {}", "2.6.0", character(), character()) ==
               nutpieR:::inline_cache_key("data {}", "2.7.0", character(), character()))
  # Argument order does not matter (sorting is part of the contract).
  expect_equal(
    nutpieR:::inline_cache_key("data {}", v, c("--O1", "--O2"), character()),
    nutpieR:::inline_cache_key("data {}", v, c("--O2", "--O1"), character())
  )
})
