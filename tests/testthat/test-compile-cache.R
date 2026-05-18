# Tests for the nutpie_compile_model() content-hashed compile cache.
#
# Strategy: cache *policy* (hit/miss/invalidation/pruning) is unit-tested
# against a stubbed compile_stan_model so we don't pay 20s per cold compile
# to verify state transitions. One real end-to-end smoke and one #include
# integration test cover the wiring between the cache layer and bridgestan.
#
# All policy tests redirect R_USER_CACHE_DIR to a tempdir so the global
# cache from helper-models.R is never touched.

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
make_compile_stub <- function(counter) {
  counter$n <- 0L
  function(stan_file, stanc_args, compile_args) {
    counter$n <- counter$n + 1L
    out <- paste0(tools::file_path_sans_ext(stan_file), "_model.so")
    file.create(out)
    out
  }
}

local_isolated_cache <- function(env = parent.frame()) {
  td <- tempfile("nutpieR-test-cache-")
  dir.create(td, recursive = TRUE)
  withr::local_envvar(c(R_USER_CACHE_DIR = td), .local_envir = env)
  withr::defer(unlink(td, recursive = TRUE), envir = env)
  # Force re-resolution of the memoized cache root under the new envvar.
  rm(list = ls(nutpieR:::.cache_state), envir = nutpieR:::.cache_state)
  td
}

test_that("cache_key folds in content, BridgeStan version, and flags", {
  v <- nutpieR:::bs_version()
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

test_that("stan_file cache: hit, edit invalidates, flag change invalidates", {
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
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

  # Editing the source changes the hash, hits a fresh cache slot.
  writeLines("// edited", stan)
  m3 <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_equal(counter$n, 2L)
  expect_false(m3$lib_path == m1$lib_path)

  # Different stanc_args invalidate even with no source change.
  nutpie_compile_model(stan_file = stan, verbose = 0L, stanc_args = "--O1")
  expect_equal(counter$n, 3L)

  # And different compile_args.
  nutpie_compile_model(stan_file = stan, verbose = 0L,
                       stanc_args = "--O1", compile_args = "STAN_THREADS=true")
  expect_equal(counter$n, 4L)
})

test_that("cache = FALSE compiles to a fresh tempdir, leaves cache untouched", {
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )

  stan <- make_temp_stan()
  on.exit(unlink(dirname(stan), recursive = TRUE), add = TRUE)

  m1 <- nutpie_compile_model(stan_file = stan, verbose = 0L, cache = FALSE)
  expect_equal(counter$n, 1L)
  # Output lives in a tempdir, NOT under the cache root.
  expect_false(startsWith(normalizePath(m1$lib_path),
                          normalizePath(nutpie_cache_dir())))

  # Persistent cache wasn't populated, so a normal-cache call must compile.
  nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_equal(counter$n, 2L)
})

test_that("inline cache: hit, miss on content/flags, clear wipes", {
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
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

test_that("stan_file and code with identical content produce identical hashes", {
  # Files-vs-code with the same content land in the *same* cache slot
  # iff the inline filename (`model.stan`) matches the stan_file's
  # relative name under its common root. file_bundle for a lone file
  # gives main = basename(file), so naming the temp file `model.stan`
  # is the only place this collision can happen.
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )

  src <- "parameters { real x; } model { x ~ normal(0, 1); }"
  stan <- make_temp_stan(src)
  on.exit(unlink(dirname(stan), recursive = TRUE), add = TRUE)

  m_file <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  m_code <- nutpie_compile_model(code = src,       verbose = 0L)
  # Same hash -> same lib_path, single compile.
  expect_equal(m_code$lib_path, m_file$lib_path)
  expect_equal(counter$n, 1L)
})

test_that("crash-safe: .so without `ok` marker is not treated as a hit", {
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )

  stan <- make_temp_stan()
  on.exit(unlink(dirname(stan), recursive = TRUE), add = TRUE)

  m1 <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_equal(counter$n, 1L)

  # Simulate an interrupted prior session: remove the ok marker. The
  # next call must recompile rather than reusing the half-finished slot.
  entry <- dirname(dirname(m1$lib_path))  # .../<hash>/src/<file>_model.so
  ok <- file.path(entry, "ok")
  expect_true(file.exists(ok))
  unlink(ok)

  nutpie_compile_model(stan_file = stan, verbose = 0L)
  expect_equal(counter$n, 2L)
  expect_true(file.exists(ok))
})

test_that("editing an #include'd file invalidates the cache", {
  local_isolated_cache()
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

  # Touch only the included file. With hash-based invalidation this
  # changes the bundle content -> new hash -> recompile.
  writeLines("// v2", prior)
  nutpie_compile_model(stan_file = main, verbose = 0L)
  expect_equal(counter$n, 2L)
})

test_that("multi-space, missing, and nested #include all invalidate properly", {
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )

  d <- tempfile("nutpieR-include-edge-")
  dir.create(d, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)

  # a -> b -> c, with deliberately weird whitespace so the regex has to
  # accept multi-space after `#include`.
  a <- file.path(d, "a.stan")
  b <- file.path(d, "b.stan")
  c <- file.path(d, "c.stan")
  writeLines("// c v1", c)
  writeLines(c("// b v1", "#include   c.stan"), b)
  writeLines(c("functions {", "#include  b.stan", "}",
               "parameters { real x; } model { x ~ normal(0, 1); }"), a)

  nutpie_compile_model(stan_file = a, verbose = 0L)
  expect_equal(counter$n, 1L)

  # Editing the depth-2 include must invalidate -- proves transitive walk.
  writeLines("// c v2", c)
  nutpie_compile_model(stan_file = a, verbose = 0L)
  expect_equal(counter$n, 2L)

  # Deleting an included file must invalidate -- a missing dep should
  # never read as "no constraint" (would silently return a stale .so).
  unlink(c)
  nutpie_compile_model(stan_file = a, verbose = 0L)
  expect_equal(counter$n, 3L)
})

test_that("commented-out #include directives are ignored", {
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )

  d <- tempfile("nutpieR-comment-test-")
  dir.create(d, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  main <- file.path(d, "main.stan")
  writeLines(c(
    "// #include ghost_line.stan",
    "/* #include ghost_block.stan */",
    "/* multi-line block",
    "   #include ghost_inside_block.stan",
    "   end */",
    "parameters { real x; } model { x ~ normal(0, 1); }"
  ), main)

  expect_equal(nutpieR:::included_files(main), character())

  nutpie_compile_model(stan_file = main, verbose = 0L)
  expect_equal(counter$n, 1L)
  nutpie_compile_model(stan_file = main, verbose = 0L)
  expect_equal(counter$n, 1L)
})

test_that("nutpie_prune_cache respects max_entries and min_age_days", {
  local_isolated_cache()
  root <- nutpie_cache_dir()

  # Synthesize a mix of "old" (eligible) and "young" (protected) entries.
  make_entry <- function(name, age_days) {
    e <- file.path(root, name)
    dir.create(file.path(e, "src"), recursive = TRUE)
    ok <- file.path(e, "ok")
    file.create(ok)
    Sys.setFileTime(ok, Sys.time() - age_days * 86400)
    e
  }
  for (i in seq_len(10)) make_entry(sprintf("old%02d", i), 30)
  for (i in seq_len(10)) make_entry(sprintf("new%02d", i),  1)

  # No-op when min_age_days excludes everything that would put us over.
  expect_equal(nutpie_prune_cache(max_entries = 5L, min_age_days = 365),
               0L)
  expect_equal(length(list.dirs(root, recursive = FALSE)), 20L)

  # With realistic params: 20 entries, cap 16, 14d min age -> 4 old
  # entries removed; all 10 young entries protected.
  expect_equal(nutpie_prune_cache(max_entries = 16L, min_age_days = 14),
               4L)
  remaining <- basename(list.dirs(root, recursive = FALSE))
  expect_equal(length(remaining), 16L)
  expect_true(all(grepl("^new", remaining[order(remaining)][1:10])))
})

test_that("entries without ok marker don't count toward cap", {
  local_isolated_cache()
  root <- nutpie_cache_dir()
  # 5 valid + 3 in-flight (no marker). Cap = 4 valid entries.
  for (i in seq_len(5)) {
    e <- file.path(root, sprintf("valid%d", i))
    dir.create(e, recursive = TRUE)
    file.create(file.path(e, "ok"))
    Sys.setFileTime(file.path(e, "ok"), Sys.time() - 30 * 86400)
  }
  for (i in seq_len(3)) {
    e <- file.path(root, sprintf("inflight%d", i))
    dir.create(e, recursive = TRUE)
  }

  # 5 valid - 4 = 1 should be removed. In-flight entries untouched.
  expect_equal(nutpie_prune_cache(max_entries = 4L, min_age_days = 1),
               1L)
  names <- basename(list.dirs(root, recursive = FALSE))
  expect_equal(sum(grepl("^valid", names)),    4L)
  expect_equal(sum(grepl("^inflight", names)), 3L)
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
