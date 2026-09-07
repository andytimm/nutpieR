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

test_that("per-entry cache lock is released after success and failure", {
  entry <- tempfile("nutpieR-cache-lock-")
  on.exit(unlink(nutpieR:::entry_lock_path(entry), recursive = TRUE), add = TRUE)

  expect_equal(nutpieR:::with_cache_entry_lock(entry, 42L), 42L)
  expect_false(dir.exists(nutpieR:::entry_lock_path(entry)))

  expect_error(nutpieR:::with_cache_entry_lock(entry, stop("expected failure")),
               "expected failure")
  expect_false(dir.exists(nutpieR:::entry_lock_path(entry)))
})

test_that("per-entry cache lock reclaims a stale lock", {
  entry <- tempfile("nutpieR-cache-lock-")
  lock <- nutpieR:::entry_lock_path(entry)
  dir.create(lock)
  on.exit(unlink(lock, recursive = TRUE), add = TRUE)

  expect_equal(
    nutpieR:::with_cache_entry_lock(
      entry, 42L, timeout_secs = 0, stale_secs = -1
    ),
    42L
  )
  expect_false(dir.exists(lock))
})

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

test_that("compile validates cache and verbose arguments", {
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

  expect_error(
    nutpie_compile_model(stan_file = stan, verbose = NA_integer_),
    "verbose"
  )
  expect_error(
    nutpie_compile_model(stan_file = stan, cache = "false", verbose = 0L),
    "cache"
  )
  expect_equal(counter$n, 0L)
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

test_that("stan_file and code with byte-identical content share a cache slot", {
  # Hashing is byte-level (readBin / writeBin), so a file written with
  # writeLines() (which appends a newline) does *not* collide with the
  # same string passed via `code =` (which has none). That divergence
  # is intentional -- byte equality is what the cache promises. We
  # demonstrate collision here by writing the file with the exact same
  # bytes as the inline code.
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )

  src <- "parameters { real x; } model { x ~ normal(0, 1); }"
  d <- tempfile("nutpieR-bytematch-")
  dir.create(d, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  stan <- file.path(d, "model.stan")
  writeBin(charToRaw(src), stan)  # byte-exact match for the inline code

  m_file <- nutpie_compile_model(stan_file = stan, verbose = 0L)
  m_code <- nutpie_compile_model(code = src,       verbose = 0L)
  # Same bytes -> same hash -> same lib_path, single compile.
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
  writeLines("real prior_mean() { return 0; }", prior)
  writeLines(c("functions {", "#include priors.stan", "}",
               "parameters { real x; } model { x ~ normal(prior_mean(), 1); }"), main)

  nutpie_compile_model(stan_file = main, verbose = 0L)
  expect_equal(counter$n, 1L)

  # Touch only the included file. With hash-based invalidation this
  # changes the bundle content -> new hash -> recompile.
  writeLines("real prior_mean() { return 1; }", prior)
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
  writeLines("real nested_mean() { return 0; }", c)
  writeLines(c("// b v1", "#include   c.stan"), b)
  writeLines(c("functions {", "#include  b.stan", "}",
               "parameters { real x; } model { x ~ normal(nested_mean(), 1); }"), a)

  nutpie_compile_model(stan_file = a, verbose = 0L)
  expect_equal(counter$n, 1L)

  # Editing the depth-2 include must invalidate -- proves transitive walk.
  writeLines("real nested_mean() { return 1; }", c)
  nutpie_compile_model(stan_file = a, verbose = 0L)
  expect_equal(counter$n, 2L)

  # A deleted dependency must not return the prior cache entry.  stanc's own
  # resolver surfaces the missing file before a cache lookup can occur.
  unlink(c)
  expect_error(nutpie_compile_model(stan_file = a, verbose = 0L),
               "include|c\\.stan|Could not find")
  expect_equal(counter$n, 2L)
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

  # The gate may conservatively false-positive on a block comment; stanc,
  # rather than an R parser, remains the authority and reports no dependency.
  expect_true(nutpieR:::has_possible_include(nutpieR:::read_dep(main)))
  expect_length(nutpieR:::resolve_included_source(main, character())$dependencies, 0L)

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

test_that("nutpie_prune_cache rejects invalid arguments before deleting", {
  local_isolated_cache()
  root <- nutpie_cache_dir()
  entry <- file.path(root, "must-survive")
  dir.create(entry, recursive = TRUE)
  file.create(file.path(entry, "ok"))

  expect_error(nutpie_prune_cache(max_entries = -1L), "max_entries")
  expect_error(nutpie_prune_cache(max_entries = 1.5), "whole number")
  expect_error(nutpie_prune_cache(max_entries = NA_real_), "finite integer")
  expect_error(nutpie_prune_cache(min_age_days = -1), "min_age_days")
  expect_error(nutpie_prune_cache(min_age_days = Inf), "finite")
  expect_error(nutpie_prune_cache(min_age_days = c(1, 2)), "single")

  expect_true(dir.exists(entry))
})

test_that("cache hit refreshes ok marker mtime (so prune treats it as LRU)", {
  # Regression: previously, a hit returned the cached model without
  # touching the marker, so a 30-day-old entry stayed evictable even
  # if the user just hit it -- auto-prune could delete a model the
  # caller still holds. Refresh on hit makes pruning LRU-ish.
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )

  src <- "parameters { real x; } model { x ~ normal(0, 1); }"
  m <- nutpie_compile_model(code = src, verbose = 0L)
  entry <- dirname(dirname(m$lib_path))
  ok <- file.path(entry, "ok")

  # Backdate the marker by 30 days; without the hit-refresh fix the
  # mtime would stay 30d old after the second compile.
  Sys.setFileTime(ok, Sys.time() - 30 * 86400)
  before <- file.info(ok)$mtime

  nutpie_compile_model(code = src, verbose = 0L)  # cache hit
  expect_equal(counter$n, 1L)
  after <- file.info(ok)$mtime
  # Allow a 1s wiggle for filesystem mtime granularity; the bump
  # should land >= 14d more recent than the backdated value.
  expect_gt(as.numeric(after) - as.numeric(before), 14 * 86400 - 1)
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

test_that("missing #include errors at compile time, not silently no-ops", {
  # Regression: read_or_empty() used to materialize missing deps as
  # empty files, so `#include missing.stan` silently compiled to a
  # no-op include. The bundle now records missing deps but does not
  # stage them, so stanc surfaces its native "could not find" error.
  skip_if_no_make()

  d <- tempfile("nutpieR-missing-inc-")
  dir.create(d, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  main <- file.path(d, "main.stan")
  prior <- file.path(d, "priors.stan")

  writeLines(c(
    "functions {",
    "#include priors.stan",
    "}",
    "parameters { real x; } model { x ~ normal(0, 1); }"
  ), main)
  writeLines("real noop_lpdf(real x) { return 0; }", prior)

  # Cold compile succeeds with the include present.
  m1 <- nutpie_compile_model(stan_file = main, verbose = 0L)
  expect_true(file.exists(m1$lib_path))

  # Delete the include, recompile -- must error rather than silently
  # producing a model where the include is a no-op. Match the error
  # against stanc's actual diagnostic so unrelated compile failures
  # don't sneak through this regression test.
  unlink(prior)
  expect_error(
    nutpie_compile_model(stan_file = main, verbose = 0L),
    "include|priors\\.stan|Could not find"
  )
})

test_that("nutpie_clear_cache only wipes the active cache root", {
  # Regression: clear_cache used to unlink both R_user_dir and
  # tempdir()/nutpieR-cache unconditionally, which could delete .so
  # files backing live nutpie_model objects in another cache root.
  # It should now only touch the resolved active root.
  td_active <- tempfile("nutpieR-active-")
  td_other  <- tempfile("nutpieR-other-")
  dir.create(td_active, recursive = TRUE)
  dir.create(td_other,  recursive = TRUE)
  withr::defer(unlink(c(td_active, td_other), recursive = TRUE))

  # Plant a marker in a "live but inactive" cache location.
  other_models <- file.path(td_other, "R", "nutpieR", "cache", "models")
  dir.create(other_models, recursive = TRUE)
  marker_other <- file.path(other_models, "some-model.so")
  file.create(marker_other)

  # Make td_active the resolved root, then plant a marker there.
  withr::with_envvar(c(R_USER_CACHE_DIR = td_active), {
    rm(list = ls(nutpieR:::.cache_state), envir = nutpieR:::.cache_state)
    root <- nutpie_cache_dir()
    marker_active <- file.path(root, "some-model.so")
    file.create(marker_active)

    nutpie_clear_cache()
    expect_false(file.exists(marker_active))
  })

  expect_true(file.exists(marker_other))
})

test_that("print.nutpie_model shows user source path, not staged copy", {
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
  m_file <- nutpie_compile_model(stan_file = stan, verbose = 0L)

  out_file <- utils::capture.output(print(m_file))
  expect_true(any(grepl(normalizePath(stan), out_file, fixed = TRUE)))
  # The staged path under the cache dir must not surface as the
  # user-facing source.
  cache_root <- nutpie_cache_dir()
  expect_false(any(grepl(cache_root, out_file, fixed = TRUE) &
                     grepl("Source", out_file)))

  m_code <- nutpie_compile_model(code = "parameters { real q; } model {}",
                                 verbose = 0L)
  out_code <- utils::capture.output(print(m_code))
  expect_true(any(grepl("<inline code>", out_code, fixed = TRUE)))
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


test_that("external include edits invalidate inline cache and change the model", {
  skip_if_no_make()
  local_isolated_cache()

  d <- tempfile("nutpieR-external-inline-")
  inc <- file.path(d, "include")
  dir.create(inc, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  center <- file.path(inc, "center.stan")
  writeLines("real center() { return 0; }", center)
  code <- paste(
    "functions {", "#include center.stan", "}",
    "parameters { real x; }", "model { x ~ normal(center(), 0.1); }",
    sep = "\n"
  )
  flags <- paste0("--include-paths=", normalizePath(inc))

  first <- nutpie_compile_model(code = code, stanc_args = flags, verbose = 0L)
  first_draws <- nutpie_sample(
    first, data = NULL, num_warmup = 80L, num_draws = 80L,
    num_chains = 1L, seed = 42L, refresh = 0L
  )
  writeLines("real center() { return 10; }", center)
  second <- nutpie_compile_model(code = code, stanc_args = flags, verbose = 0L)
  second_draws <- nutpie_sample(
    second, data = NULL, num_warmup = 80L, num_draws = 80L,
    num_chains = 1L, seed = 42L, refresh = 0L
  )

  expect_false(identical(first$lib_path, second$lib_path))
  expect_lt(mean(as.numeric(first_draws)), 1)
  expect_gt(mean(as.numeric(second_draws)), 9)
  # The unchanged expanded source/dependency bytes retain the ordinary hit.
  expect_identical(
    nutpie_compile_model(code = code, stanc_args = flags, verbose = 0L)$lib_path,
    second$lib_path
  )
})

test_that("compiler resolution handles nested includes from the main root", {
  skip_if_no_make()
  local_isolated_cache()

  d <- tempfile("nutpieR-nested-main-root-")
  dir.create(file.path(d, "sub"), recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  writeLines("real center() { return 0; }", file.path(d, "center.stan"))
  writeLines("#include center.stan", file.path(d, "sub", "f.stan"))
  main <- file.path(d, "main.stan")
  writeLines(c(
    "functions {", "#include sub/f.stan", "}",
    "parameters { real x; }", "model { x ~ normal(center(), 1); }"
  ), main)

  # This also exercises the fresh staging route: no source-tree layout is
  # assumed after stanc has expanded the include tree.
  fresh <- nutpie_compile_model(stan_file = main, cache = FALSE, verbose = 0L)
  expect_true(file.exists(fresh$lib_path))

  cached <- nutpie_compile_model(stan_file = main, verbose = 0L)
  cached_draws <- nutpie_sample(
    cached, data = NULL, num_warmup = 80L, num_draws = 80L,
    num_chains = 1L, seed = 42L, refresh = 0L
  )
  writeLines("real center() { return 2; }", file.path(d, "center.stan"))
  changed <- nutpie_compile_model(stan_file = main, verbose = 0L)
  changed_draws <- nutpie_sample(
    changed, data = NULL, num_warmup = 80L, num_draws = 80L,
    num_chains = 1L, seed = 42L, refresh = 0L
  )
  expect_false(identical(cached$lib_path, changed$lib_path))
  expect_lt(mean(as.numeric(cached_draws)), 1)
  expect_gt(mean(as.numeric(changed_draws)), 1)
})

test_that("stanc include search order is retained when resolving dependencies", {
  skip_if_no_make()
  # Exercise literal tildes on every platform, as in Windows RUNNER~1 paths.
  d <- tempfile("nutpieR~1-include-order-")
  a <- file.path(d, "a")
  b <- file.path(d, "b")
  dir.create(a, recursive = TRUE)
  dir.create(b, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  writeLines("real center() { return 1; }", file.path(a, "center.stan"))
  writeLines("real center() { return 2; }", file.path(b, "center.stan"))
  main <- file.path(d, "main.stan")
  writeLines(c("functions {", "#include center.stan", "}",
               "parameters { real x; } model { x ~ normal(center(), 1); }"), main)

  first <- nutpieR:::resolve_included_source(
    main, c(paste0("--include-paths=", a), paste0("--include-paths=", b))
  )
  second <- nutpieR:::resolve_included_source(
    main, c(paste0("--include-paths=", b), paste0("--include-paths=", a))
  )
  expect_type(first, "list")
  expect_type(second, "list")
  expect_identical(first$dependencies[[1L]]$path, normalizePath(file.path(a, "center.stan"), winslash = "/"))
  expect_identical(second$dependencies[[1L]]$path, normalizePath(file.path(b, "center.stan"), winslash = "/"))
})


test_that("untrackable stanc output modes bypass the persistent include cache", {
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    compile_stan_model = make_compile_stub(counter),
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )
  code <- paste("functions {", "#include external.stan", "}",
                "parameters { real x; } model {}", sep = "\n")
  expect_warning(
    first <- nutpie_compile_model(code = code, stanc_args = "--auto-format", verbose = 0L),
    "without the persistent cache"
  )
  expect_warning(
    second <- nutpie_compile_model(code = code, stanc_args = "--auto-format", verbose = 0L),
    "without the persistent cache"
  )
  expect_equal(counter$n, 2L)
  expect_false(identical(first$lib_path, second$lib_path))
  expect_false(startsWith(normalizePath(first$lib_path), normalizePath(nutpie_cache_dir())))
})


test_that("the include gate has no mid-line directive false negative", {
  expect_true(nutpieR:::has_possible_include(
    charToRaw("functions { #include f.stan\n} parameters { real x; } model {}")
  ))
})

test_that("make compiler overrides bypass tracking before stanc and retain file root", {
  local_isolated_cache()
  counter <- new.env(parent = emptyenv())
  compile_stub <- make_compile_stub(counter)
  testthat::local_mocked_bindings(
    compile_stan_model = function(stan_file, stanc_args, compile_args) {
      counter$source_lock_seen <- any(grepl(
        "^\\.nutpieR-untracked-.*\\.lock$",
        list.files(dirname(stan_file), all.files = TRUE)
      ))
      compile_stub(stan_file, stanc_args, compile_args)
    },
    bridgestan_stanc_path = function() stop("resolver must not run"),
    bs_version = function() "TEST.0",
    bridgestan_version = function() "TEST.0",
    .package = "nutpieR"
  )
  d <- tempfile("nutpieR-untracked-root-")
  dir.create(d)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  main <- file.path(d, "main.stan")
  writeLines(c("functions { #include f.stan }", "parameters { real x; } model {}"), main)

  expect_warning(
    model <- nutpie_compile_model(
      stan_file = main, compile_args = "STANC=custom-stanc", verbose = 0L
    ),
    "without the persistent cache"
  )
  expect_equal(counter$n, 1L)
  # The fallback compiled the user source itself (not a guessed staged tree),
  # then copied its result to a distinct path for safe dlopen behavior.
  expect_identical(
    normalizePath(model$staged_source),
    normalizePath(main)
  )
  expect_false(startsWith(normalizePath(model$lib_path), normalizePath(d)))
  expect_true(counter$source_lock_seen)
})


test_that("make-local and makefile overrides conservatively disable tracking", {
  d <- tempfile("nutpieR-make-local-")
  dir.create(file.path(d, "bin"), recursive = TRUE)
  dir.create(file.path(d, "make"), recursive = TRUE)
  fake_stanc <- file.path(d, "bin", "stanc")
  file.create(fake_stanc)
  writeLines("# arbitrary local make customization", file.path(d, "make", "local"))
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  expect_true(nutpieR:::stanc_make_override_present(fake_stanc))
  # These forms are detected before the resolver could invoke bundled stanc.
  for (arg in c("-f", "-fother.mk", "STANC:=custom", "STANC+=custom",
                "STANC?=custom", "STANCFLAGS:=--O1", "--eval=STANC=custom",
                "-ESTANC=custom")) {
    expect_false(nutpieR:::stanc_tracking_supported(character(), arg))
  }
})

test_that("untrackable file source paths with spaces fail before compilation", {
  d <- tempfile("nutpieR source space ")
  dir.create(d)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  main <- file.path(d, "main.stan")
  writeLines(c("functions { #include f.stan }", "parameters { real x; } model {}"), main)
  expect_error(
    suppressWarnings(nutpie_compile_model(
      stan_file = main, compile_args = "STANC=custom", verbose = 0L
    )),
    "writable and have no spaces"
  )
})

test_that("a newly shadowing main-root include invalidates a file cache entry", {
  skip_if_no_make()
  local_isolated_cache()
  d <- tempfile("nutpieR-new-shadow-")
  external <- file.path(d, "external")
  dir.create(external, recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  writeLines("real center() { return 0; }", file.path(external, "center.stan"))
  main <- file.path(d, "main.stan")
  writeLines(c(
    "functions { #include center.stan }", "parameters { real x; }",
    "model { x ~ normal(center(), 0.1); }"
  ), main)
  flags <- paste0("--include-paths=", normalizePath(external))
  before <- nutpie_compile_model(stan_file = main, stanc_args = flags, verbose = 0L)
  before_draws <- nutpie_sample(
    before, data = NULL, num_warmup = 80L, num_draws = 80L,
    num_chains = 1L, seed = 42L, refresh = 0L
  )
  # BridgeStan prepends the main source root before user paths.  A newly
  # present file there must supersede the old external resolution and cache.
  writeLines("real center() { return 3; }", file.path(d, "center.stan"))
  after <- nutpie_compile_model(stan_file = main, stanc_args = flags, verbose = 0L)
  after_draws <- nutpie_sample(
    after, data = NULL, num_warmup = 80L, num_draws = 80L,
    num_chains = 1L, seed = 42L, refresh = 0L
  )
  expect_false(identical(before$lib_path, after$lib_path))
  expect_lt(mean(as.numeric(before_draws)), 1)
  expect_gt(mean(as.numeric(after_draws)), 2)
})
