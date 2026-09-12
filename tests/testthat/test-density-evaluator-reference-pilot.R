# Isolate checker orchestration from the native DLL for the fast test tier.
pilot_checker <- function() {
  env <- new.env(parent = environment(nutpie_validate_density_evaluator))
  validate <- nutpie_validate_density_evaluator
  environment(validate) <- env
  env$evaluator_check_dimension <- function(model) 2L
  env$pilot_calls <- 0L
  env$batches <- list()
  env$evaluator_check_pilot <- function(model, num_points, seed) {
    env$pilot_calls <- env$pilot_calls + 1L
    list(points = matrix(seq_len(num_points * 2L), ncol = 2L),
         pilot = list(settings = list(num_draws = num_points, seed = seed),
                      diagnostics = list(marker = "reference only"), elapsed_seconds = 0.1))
  }
  env$evaluator_check_evaluate <- function(model, points) {
    env$batches[[length(env$batches) + 1L]] <- points
    values <- lapply(seq_len(nrow(points)), function(i)
      list(status = "ok", message = "", logp = -sum(points[i, ]^2) / 2,
           gradient = -points[i, ]))
    list(reference = values, evaluator = values)
  }
  list(validate = validate, env = env)
}

pilot_rng_restore <- function() {
  had <- exists(".Random.seed", .GlobalEnv, inherits = FALSE)
  old <- if (had) get(".Random.seed", .GlobalEnv) else NULL
  function() {
    if (had) assign(".Random.seed", old, .GlobalEnv)
    else if (exists(".Random.seed", .GlobalEnv, inherits = FALSE))
      rm(".Random.seed", envir = .GlobalEnv)
  }
}

test_that("default random and explicit points never start a reference pilot", {
  x <- pilot_checker()
  x$env$evaluator_check_pilot <- function(...) stop("unexpected pilot")
  default <- x$validate(NULL, num_points = 3L, seed = 9L)
  random <- x$validate(NULL, num_points = 3L, seed = 9L, method = "random")
  expect_identical(default$points, random$points)
  expect_identical(default$method, "random")
  expect_null(default$pilot)
  points <- rbind(c(1, 2), c(3, 4))
  explicit <- x$validate(NULL, points = points)
  expect_identical(explicit$points, points)
  expect_null(explicit$pilot)
  expect_true(all(explicit$point_source == "explicit"))
  expect_error(x$validate(NULL, points = points, method = "reference"), "points|reference")
  expect_error(x$validate(NULL, method = "unknown"), "arg|method")
})

test_that("reference points and four broad probes use one repeatability batch", {
  x <- pilot_checker()
  result <- x$validate(NULL, method = "reference", num_points = 3L, seed = 19L, radius = 5)
  expect_identical(x$env$pilot_calls, 1L)
  expect_identical(result$method, "reference")
  expect_identical(result$status, "pass")
  expect_true(result$advisory)
  expect_equal(dim(result$points), c(7L, 2L))
  expect_equal(result$points[1:3, ], matrix(1:6, ncol = 2))
  expect_true(all(abs(result$points[4:7, ]) <= 5))
  expect_length(x$env$batches, 1L)
  expect_equal(x$env$batches[[1L]],
               rbind(result$points, result$points[c(1L, 2L, 1L), ]))
  expect_identical(result$pilot$diagnostics$marker, "reference only")
  expect_equal(result$pilot$settings$num_draws, 3L)
  expect_equal(result$pilot$elapsed_seconds, 0.1)
  again <- x$validate(NULL, method = "reference", num_points = 3L, seed = 19L, radius = 5)
  expect_identical(result$points, again$points)
  changed <- x$validate(NULL, method = "reference", num_points = 3L, seed = 20L, radius = 5)
  expect_false(identical(result$points[4:7, ], changed$points[4:7, ]))
})

test_that("reference generation restores present and absent RNG state on every exit", {
  restore <- pilot_rng_restore()
  on.exit(restore(), add = TRUE)
  for (had in c(TRUE, FALSE)) {
    if (had) { set.seed(817); old <- .Random.seed }
    else if (exists(".Random.seed", .GlobalEnv, inherits = FALSE))
      rm(".Random.seed", envir = .GlobalEnv)
    check_rng <- function() {
      expect_identical(exists(".Random.seed", .GlobalEnv, inherits = FALSE), had)
      if (had) expect_identical(get(".Random.seed", .GlobalEnv), old)
    }
    x <- pilot_checker()
    x$validate(NULL, method = "reference", num_points = 2L)
    check_rng()
    x$env$evaluator_check_pilot <- function(...) {
      stats::runif(2)
      stop("pilot deliberately failed")
    }
    expect_error(x$validate(NULL, method = "reference"), "pilot deliberately failed")
    expect_length(x$env$batches, 1L) # no fallback evaluation
    check_rng()
    x <- pilot_checker()
    x$env$evaluator_check_evaluate <- function(...) stop("batch deliberately failed")
    expect_error(x$validate(NULL, method = "reference"), "batch deliberately failed")
    check_rng()
  }
})

test_that("invalid reference inputs fail before starting a pilot", {
  x <- pilot_checker()
  for (seed in list(NA_real_, -1, 1.5, Inf, NULL, .Machine$integer.max + 1))
    expect_error(x$validate(NULL, method = "reference", seed = seed), "seed")
  for (n in c(0, 1, 2.5, Inf, NA_real_))
    expect_error(x$validate(NULL, method = "reference", num_points = n), "num_points")
  expect_identical(x$env$pilot_calls, 0L)
})


test_that("mixed point groups cannot hide inconclusive or failed probes", {
  for (bad_source in c("reference", "random")) {
    for (bad_status in c("inconclusive", "fail")) {
      x <- pilot_checker()
      original <- x$env$evaluator_check_evaluate
      row <- if (bad_source == "reference") 1L else 4L
      x$env$evaluator_check_evaluate <- function(model, points) {
        values <- original(model, points)
        if (bad_status == "inconclusive")
          values$reference[[row]] <- list(status = "domain", message = "outside domain")
        else values$evaluator[[row]]$gradient[2L] <- 1000
        values
      }
      result <- x$validate(NULL, method = "reference", num_points = 3L)
      expect_identical(result$point_source, c(rep("reference", 3L), rep("random", 4L)))
      expect_identical(result$comparisons$point_source, result$point_source)
      expect_identical(result$status, bad_status)
      group <- result$groups[result$groups$point_source == bad_source, ]
      other <- result$groups[result$groups$point_source != bad_source, ]
      expect_identical(group$status, bad_status)
      expect_equal(group[[bad_status]], 1L)
      expect_equal(group$pass, group$n - 1L)
      expect_identical(other$status, "pass")
      expect_equal(sum(result$groups$n), 7L)
    }
  }
})

test_that("incomplete or nonfinite pilot rows are discarded without fallback", {
  for (points in list(matrix(1, 1, 2), matrix(1, 3, 1),
                      matrix(NA_real_, 3, 2), matrix(Inf, 3, 2), 1:6)) {
    x <- pilot_checker()
    x$env$evaluator_check_pilot <- function(...) list(points = points, pilot = list())
    expect_error(x$validate(NULL, method = "reference", num_points = 3L),
                 "incomplete|invalid|discarded")
    expect_length(x$env$batches, 0L)
  }
})


test_that("pilot helper sends only the reference owner and records fixed settings", {
  helper <- evaluator_check_pilot
  env <- new.env(parent = environment(helper))
  environment(helper) <- env
  owner <- new.env()
  env$bs_ndim_unc <- function(bs_ptr) {
    expect_identical(bs_ptr, owner)
    2L
  }
  env$bs_reference_pilot <- function(bs_ptr, num_points, seed) {
    expect_identical(bs_ptr, owner)
    expect_identical(num_points, 3L)
    expect_identical(seed, 42L)
    list(points = matrix(1:6, ncol = 2), diagnostics = list(marker = TRUE),
         sampler_config = "native config")
  }
  result <- helper(list(bs_ptr = owner, evaluator_ptr = "must not be used"), 3L, 42L)
  expect_equal(result$points, matrix(1:6, ncol = 2))
  expect_identical(result$pilot$settings, list(num_chains = 1L, cores = 1L,
    num_warmup = 200L, num_draws = 3L, seed = 42L, target_accept = 0.8,
    max_treedepth = 10L, adaptation = "diag", init_radius = 2))
  expect_identical(result$pilot$diagnostics, list(marker = TRUE))
  expect_identical(result$pilot$sampler_config, "native config")
  expect_true(is.finite(result$pilot$elapsed_seconds))
  expect_gte(result$pilot$elapsed_seconds, 0)
})

# This function is written to a temporary script, never run in the test process.
pilot_native_regression <- function() {
  library(nutpieR)
  fixture <- commandArgs(trailingOnly = TRUE)[[1L]]
  work <- tempfile("byold-pilot-")
  dir.create(work)
  old <- setwd(work)
  on.exit({ setwd(old); unlink(work, recursive = TRUE) }, add = TRUE)
  stopifnot(file.copy(file.path(fixture, "gaussian.c"), work))
  include <- system.file("include", package = "nutpieR")
  stopifnot(nzchar(include))
  # Every evaluator evaluation fails and poisons its outputs. A successful pilot
  # therefore cannot have used this producer for its trajectory or expansion.
  Sys.setenv(PKG_CPPFLAGS = paste0('-I"', include, '" -DFIXTURE_MODE=8'))
  status <- system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", "gaussian.c"),
                    stdout = "build.log", stderr = "build.log", timeout = 90)
  if (status != 0L) stop(paste(readLines("build.log"), collapse = "\n"))
  lib <- normalizePath(paste0("gaussian", .Platform$dynlib.ext))
  reference <- nutpie_compile_model(code = '
    data { int n; real mu; real sigma; }
    transformed data { real centre = 1000 + normal_rng(0, 10); }
    parameters { vector[n] x; }
    model { x ~ normal(centre, sigma); }
    generated quantities { real forbidden = 0; reject("pilot ran generated quantities"); }
  ')
  bound <- nutpie_attach_density_evaluator(reference, lib, list(n = 2L, mu = 0, sigma = 0.01))
  origin <- nutpieR:::bs_evaluate(bound$bs_ptr, list(c(0, 0)))[[1L]]
  centre <- origin$gradient[1L] * 0.0001
  pilot <- function(seed) nutpieR:::evaluator_check_pilot(bound, 8L, seed)
  set.seed(817)
  rng <- .Random.seed
  a <- pilot(42L)
  b <- pilot(42L)
  c <- pilot(43L)
  stopifnot(identical(.Random.seed, rng), identical(dim(a$points), c(8L, 2L)),
            all(is.finite(a$points)), identical(a$points, b$points),
            !identical(a$points, c$points), is.list(a$pilot$settings),
            !is.null(a$pilot$diagnostics), is.finite(a$pilot$elapsed_seconds))
  stopifnot(all(abs(a$points - centre) < 0.1), all(abs(c$points - centre) < 0.1))
  # The same bound reference and its transformed-data RNG snapshot survive
  # changes to the pilot seed. No reopening with the sampling seed is allowed.
  for (p in list(a, c)) {
    values <- nutpieR:::bs_evaluate(bound$bs_ptr,
      lapply(seq_len(nrow(p$points)), function(i) p$points[i, ]))
    for (i in seq_along(values))
      stopifnot(max(abs(values[[i]]$gradient + (p$points[i, ] - centre) / 0.0001)) < 1e-7)
  }
  after <- nutpieR:::bs_evaluate(bound$bs_ptr, list(c(0, 0)))[[1L]]
  stopifnot(identical(origin, after))
  report <- nutpie_validate_density_evaluator(bound, method = "reference", num_points = 8L, seed = 42L)
  stopifnot(report$status == "fail", nrow(report$points) == 12L,
            all(report$comparisons$status == "fail"), !is.null(report$pilot))
  # A dead evaluator handle is irrelevant to the reference-only native entrypoint.
  dead <- unserialize(serialize(bound, NULL))
  native <- nutpieR:::bs_reference_pilot(bound$bs_ptr, 8L, 42L)
  stopifnot(!is.null(native$points), !is.null(native$diagnostics),
            !is.null(native$sampler_config))
  error <- function(expr) {
    e <- tryCatch({ force(expr); NULL }, error = identity)
    stopifnot(inherits(e, "error"), grepl("dead|pointer|reattach|rebind", conditionMessage(e),
                                        ignore.case = TRUE))
  }
  error(nutpieR:::bs_reference_pilot(dead$bs_ptr, 8L, 42L))
  error(nutpie_validate_density_evaluator(dead, method = "reference", num_points = 8L))
  # Keep a live reference with the serialized (dead) producer handle.
  reference_only <- list(bs_ptr = bound$bs_ptr, evaluator_ptr = dead$evaluator_ptr)
  only_reference <- nutpieR:::evaluator_check_pilot(reference_only, 8L, 42L)
  stopifnot(identical(a$points, only_reference$points))
  # Simplex dimension differs from constrained output, and the target is
  # asymmetric. Retained coordinates must be unconstrained, not a truncated
  # constrained vector or TP/GQ columns.
  simplex <- nutpie_compile_model(code = '
    parameters { simplex[3] x; }
    model { x ~ dirichlet([2, 20, 3]\'); }
    generated quantities { real forbidden = 0; reject("unexpected GQ"); }
  ')
  simplex_ptr <- nutpieR:::bs_open(simplex$lib_path, "{}", 0L)
  s <- nutpieR:::evaluator_check_pilot(list(bs_ptr = simplex_ptr), 24L, 42L)
  stopifnot(identical(dim(s$points), c(24L, 2L)), all(is.finite(s$points)),
            any(s$points < 0 | s$points > 1))
  ordered <- nutpie_compile_model(code = '
    parameters { real first; real second; }
    model { first ~ normal(10, 0.1); second ~ normal(-10, 0.1); }
  ')
  ordered_ptr <- nutpieR:::bs_open(ordered$lib_path, "{}", 0L)
  o <- nutpieR:::evaluator_check_pilot(list(bs_ptr = ordered_ptr), 8L, 42L)
  stopifnot(identical(dim(o$points), c(8L, 2L)),
            all(abs(o$points[, 1L] - 10) < 1), all(abs(o$points[, 2L] + 10) < 1))
  impossible <- nutpie_compile_model(code = '
    parameters { real x; }
    model { target += negative_infinity(); }
  ')
  impossible_ptr <- nutpieR:::bs_open(impossible$lib_path, "{}", 0L)
  failure <- tryCatch(nutpieR:::evaluator_check_pilot(list(bs_ptr = impossible_ptr), 2L, 42L),
                      error = identity)
  stopifnot(inherits(failure, "error"),
            grepl("Reference pilot failed", conditionMessage(failure)))
  stopifnot(file.copy(file.path(fixture, "gaussian.c"), "gaussian_good.c"))
  Sys.setenv(PKG_CPPFLAGS = paste0('-I"', include, '" -DFIXTURE_MODE=0'))
  status <- system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", "gaussian_good.c"),
                    stdout = "good-build.log", stderr = "good-build.log", timeout = 90)
  if (status != 0L) stop(paste(readLines("good-build.log"), collapse = "\n"))
  good_ref <- nutpie_compile_model(file.path(fixture, "gaussian.stan"))
  good <- nutpie_attach_density_evaluator(good_ref,
    normalizePath(paste0("gaussian_good", .Platform$dynlib.ext)),
    list(n = 2L, mu = 1, sigma = 2))
  draw <- function() nutpie_sample(good, num_draws = 8L, num_warmup = 200L,
                                   num_chains = 1L, cores = 1L, seed = 42L, progress = "none")
  before <- draw()
  check <- nutpie_validate_density_evaluator(good, method = "reference", num_points = 8L, seed = 43L)
  after <- draw()
  stopifnot(check$status == "pass", identical(as.numeric(before), as.numeric(after)))
  cat("REFERENCE PILOT REGRESSIONS PASSED\n")
}

test_that("native reference pilots are isolated and bounded", {
  skip_if(Sys.getenv("NUTPIER_RUN_BYOLD_TESTS") != "1",
          "set NUTPIER_RUN_BYOLD_TESTS=1 after a release install")
  fixture <- normalizePath(test_path("byold-v1"), mustWork = TRUE)
  script <- tempfile("byold-pilot-", fileext = ".R")
  log <- tempfile("byold-pilot-", fileext = ".log")
  on.exit(unlink(script), add = TRUE)
  writeLines(c("run <-", deparse(pilot_native_regression), "run()"), script)
  status <- system2(file.path(R.home("bin"), "Rscript"),
                    c(shQuote(script), shQuote(fixture)),
                    stdout = log, stderr = log, timeout = 300)
  output <- paste(readLines(log, warn = FALSE), collapse = "\n")
  expect_equal(status, 0L, info = paste(log, output))
  expect_match(output, "REFERENCE PILOT REGRESSIONS PASSED")
})
