# Tests for the nutpie_compile_model() compile cache.
#
# Strategy: cache *policy* (hit/miss/invalidation rules) is unit-tested
# against a stubbed compile_stan_model so we don't pay 20s per cold compile
# to verify state transitions. One real end-to-end smoke and one #include
# integration test cover the wiring between the cache layer and bridgestan.

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

# Stub for compile_stan_model that just touches the expected output path
# (mirroring bridgestan's "<base>_model.so" naming) and bumps a counter.
# Usage: counter <- new.env(); ... ; expect_equal(counter$n, k).
make_compile_stub <- function(counter) {
  counter$n <- 0L
  function(stan_file, stanc_args, compile_args) {
    counter$n <- counter$n + 1L
    out <- paste0(tools::file_path_sans_ext(stan_file), "_model.so")
    file.create(out)
    out
  }
}

test_that("inline cache key folds in content, BridgeStan version, and flags", {
  v <- nutpieR:::bridgestan_version()
  k_base <- nutpieR:::inline_cache_key("data {}", v, character(), character())

  expect_false(k_base == nutpieR:::inline_cache_key(
    "data { int N; }", v, character(), character()))
  expect_false(k_base == nutpieR:::inline_cache_key(
    "data {}", v, "--O1", character()))
  expect_false(k_base == nutpieR:::inline_cache_key(
    "data {}", v, character(), "STANCFLAGS=foo"))
  expect_false(
    nutpieR:::inline_cache_key("data {}", "2.6.0", character(), character()) ==
    nutpieR:::inline_cache_key("data {}", "2.7.0", character(), character())
  )
  # Argument order is meaningful (override-style flags) -- different orders
  # land in different cache slots rather than silently coalescing.
  expect_false(
    nutpieR:::inline_cache_key("data {}", v, c("--O1", "--O2"), character()) ==
    nutpieR:::inline_cache_key("data {}", v, c("--O2", "--O1"), character())
  )
})

test_that("stan_file cache: hit, edit invalidates, flag change invalidates, cache=FALSE forces", {
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    # Mock the memoized wrapper too, otherwise a real value cached at
    # session start (via helper-models.R) would shadow the stub.
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )

  stan <- make_temp_stan()
  on.exit(unlink(dirname(stan), recursive = TRUE), add = TRUE)

  # Cold compile.
  m1 <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_equal(counter$n, 1L)
  expect_true(file.exists(m1$lib_path))

  # Warm hit -- same path, no compile.
  m2 <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_equal(counter$n, 1L)
  expect_equal(m2$lib_path, m1$lib_path)

  # Edit + mtime bump invalidates.
  Sys.sleep(1.1)
  writeLines("// edited", stan)
  Sys.setFileTime(stan, Sys.time())
  nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_equal(counter$n, 2L)

  # Different stanc_args invalidate even with no source change.
  nutpie_compile_model(stan_file = stan, verbose = 0L, stanc_args = "--O1")
  expect_equal(counter$n, 3L)

  # And different compile_args.
  nutpie_compile_model(stan_file = stan, verbose = 0L,
                       stanc_args = "--O1", compile_args = "STAN_THREADS=true")
  expect_equal(counter$n, 4L)

  # cache = FALSE always recompiles, then sidecar is updated so the next
  # default-args call hits the freshly-rebuilt artifact.
  nutpie_compile_model(stan_file = stan, verbose = 0L, cache = FALSE)
  expect_equal(counter$n, 5L)
  nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_equal(counter$n, 5L)

  # Sidecar is colocated with the artifact and matches what we wrote last.
  meta <- nutpieR:::read_cache_meta(stan)
  expect_equal(meta$bs_version, "TEST.0")
  expect_equal(meta$stanc_args, character())
  expect_equal(meta$compile_args, character())
})

test_that("inline cache: hit, miss on content/flags, clear wipes", {
  counter <- new.env(parent = emptyenv())
  td <- tempfile("nutpieR-test-cache-")
  dir.create(td, recursive = TRUE)
  withr::local_envvar(c(R_USER_CACHE_DIR = td))
  withr::defer(unlink(td, recursive = TRUE))

  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    # Mock the memoized wrapper too, otherwise a real value cached at
    # session start (via helper-models.R) would shadow the stub.
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )

  src_a <- "parameters { real a; } model { a ~ normal(0, 1); }"
  src_b <- "parameters { real b; } model { b ~ normal(0, 2); }"

  m1 <- nutpie_compile_model(code = src_a, verbose = 0L)
  expect_equal(counter$n, 1L)

  m2 <- nutpie_compile_model(code = src_a, verbose = 0L)
  expect_equal(counter$n, 1L)
  expect_equal(m2$lib_path, m1$lib_path)

  m3 <- nutpie_compile_model(code = src_b, verbose = 0L)
  expect_equal(counter$n, 2L)
  expect_false(m3$lib_path == m1$lib_path)

  m4 <- nutpie_compile_model(code = src_a, verbose = 0L, stanc_args = "--O1")
  expect_equal(counter$n, 3L)
  expect_false(m4$lib_path == m1$lib_path)

  nutpie_clear_cache()
  nutpie_compile_model(code = src_a, verbose = 0L)
  expect_equal(counter$n, 4L)
})

test_that("end-to-end smoke: cold compile + warm hit returns a loadable model", {
  skip_if_no_make()

  stan <- testthat::test_path("test_models", "cache_smoke.stan")

  m1 <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_s3_class(m1, "nutpie_model")
  expect_true(file.exists(m1$lib_path))

  warm <- system.time(
    m2 <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  )[["elapsed"]]
  expect_equal(m2$lib_path, m1$lib_path)
  expect_lt(warm, 2)

  # The .so is actually loadable through bridgestan.
  handle <- nutpieR:::bs_open(m1$lib_path, "{}", 1L)
  expect_false(is.null(handle))
})

test_that("stan_file with relative #include compiles", {
  skip_if_no_make()

  d <- tempfile("nutpieR-include-test-")
  dir.create(d, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)

  writeLines(
    "real my_prior_lpdf(real x) { return normal_lpdf(x | 0, 1); }",
    file.path(d, "priors.stan")
  )
  writeLines(c(
    "functions {",
    "#include priors.stan",
    "}",
    "parameters { real x; }",
    "model { target += my_prior_lpdf(x); }"
  ), file.path(d, "main.stan"))

  m <- nutpie_compile_model(stan_file = file.path(d, "main.stan"),
                            verbose = 0L)
  expect_true(file.exists(m$lib_path))
})

test_that("editing an #include'd file invalidates the cache", {
  # Mocked so we can verify the invalidation policy without paying for two
  # real compiles; the integration test above already proves #include
  # actually compiles end-to-end.
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )

  d <- tempfile("nutpieR-include-invalidation-")
  dir.create(d, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  prior <- file.path(d, "priors.stan")
  main <- file.path(d, "main.stan")
  writeLines("// v1", prior)
  writeLines(c("functions {", "#include priors.stan", "}",
               "parameters { real x; } model { x ~ normal(0, 1); }"), main)

  nutpie_compile_model(stan_file = main, verbose = 0L)
  expect_equal(counter$n, 1L)

  # Touch only the included file. main.stan's mtime stays put -- a
  # main-file-only check would miss this and return the stale .so.
  Sys.sleep(1.1)
  writeLines("// v2", prior)
  Sys.setFileTime(prior, Sys.time())
  nutpie_compile_model(stan_file = main, verbose = 0L)
  expect_equal(counter$n, 2L)
})

test_that("read-only stan_file directory + #include errors instead of falling back", {
  # The inline-cache fallback flattens the .stan to a string and loses the
  # source dirname, so #include can't resolve. Refuse rather than silently
  # produce a broken compile.
  testthat::local_mocked_bindings(
    dir_writable = function(path) FALSE,
    .package = "nutpieR"
  )

  d <- tempfile("nutpieR-readonly-include-")
  dir.create(d, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  writeLines("// noop", file.path(d, "priors.stan"))
  writeLines(c("functions {", "#include priors.stan", "}",
               "parameters { real x; } model { x ~ normal(0, 1); }"),
             file.path(d, "main.stan"))

  expect_error(
    nutpie_compile_model(stan_file = file.path(d, "main.stan"), verbose = 0L),
    "uses `#include`"
  )
})
