# These tests exercise report logic without loading any producer library.
checker_ok <- function(q) list(status = "ok", message = "", logp = -sum(q^2) / 2, gradient = -q)
checker_tolerances <- list(logp_atol = 1e-8, logp_rtol = 1e-6,
                          gradient_atol = 1e-8, gradient_rtol = 1e-6)
checker_case <- function(change = identity, reference_change = identity,
                         points = rbind(c(1, 2), c(2, 3))) {
  order <- c(seq_len(nrow(points)), 1L, 2L, 1L)
  r <- lapply(order, function(i) checker_ok(points[i, ]))
  evaluator_check_report(points, list(reference = reference_change(r), evaluator = change(r)),
                      checker_tolerances, c(1L, 2L, 1L), NULL)
}

test_that("strict agreement reports all coordinates and mandatory repeatability", {
  report <- checker_case()
  expect_s3_class(report, "nutpie_density_evaluator_check")
  expect_named(report, c("status", "advisory", "convention", "points", "seed",
                         "tolerances", "comparisons", "gradients", "repeatability",
                         "constant_offset", "counts", "untested"))
  expect_named(report$comparisons, c("point", "status", "reason", "reference_status",
                                      "evaluator_status", "reference_message",
                                      "evaluator_message", "logp_error", "logp_pass",
                                      "gradient_pass"))
  expect_named(report$gradients,
               c("point", "coordinate", "reference", "evaluator", "error", "pass"))
  expect_identical(report$status, "pass")
  expect_true(report$advisory)
  expect_equal(nrow(report$gradients), 4L)
  expect_identical(report$repeatability$status, "pass")
  expect_identical(report$convention, list(propto = TRUE, jacobian = TRUE))
})

test_that("constant offsets are diagnosed but fail strict agreement", {
  report <- checker_case(function(x) lapply(x, function(v) { v$logp <- v$logp + 10; v }))
  expect_identical(report$status, "fail")
  expect_identical(report$constant_offset$status, "possible_constant_offset")
  expect_equal(report$constant_offset$offset, 10)
})

test_that("a mismatch in the last gradient coordinate cannot hide", {
  report <- checker_case(function(x) { x[[2]]$gradient[2] <- 0; x })
  expect_identical(report$status, "fail")
  expect_equal(subset(report$gradients, !pass)$coordinate, 2L)
})

test_that("length, finite and failed-status output checks do not consume poison", {
  mutations <- list(
    function(v) { v$gradient <- v$gradient[1]; v },
    function(v) { v$gradient[2] <- NaN; v },
    function(v) { v$logp <- Inf; v },
    function(v) { v$status <- "fatal"; v$logp <- "poison"; v$gradient <- NULL; v },
    function(v) { v$status <- "domain"; v$logp <- NULL; v })
  for (mutation in mutations) {
    report <- checker_case(function(x) { x[[1]] <- mutation(x[[1]]); x })
    expect_identical(report$status, "fail")
    expect_identical(report$comparisons$reason[1], "evaluator_invalid")
  }
})

test_that("invalid references are inconclusive, including zero comparable points", {
  reject <- function(x) lapply(x, function(v) list(status = "domain", message = "outside domain"))
  report <- checker_case(reference_change = reject)
  expect_identical(report$status, "inconclusive")
  expect_true(all(report$comparisons$reason == "reference_invalid"))
  expect_identical(checker_case(change = reject, reference_change = reject)$status, "inconclusive")
  bad <- function(x) { x[[1]] <- list(status = "fatal", message = "bad"); x }
  expect_identical(checker_case(change = bad, reference_change = reject)$status, "fail")
})

test_that("q1 q2 q1 detects stale mutable workspace results", {
  report <- checker_case(function(x) { x[[5]]$gradient[1] <- 100; x })
  expect_true(all(report$comparisons$status == "pass"))
  expect_identical(report$repeatability$status, "fail")
  expect_identical(report$status, "fail")
})

test_that("relative and absolute tolerance scale safely", {
  expect_true(evaluator_check_close(1e10 + 1, 1e10, 0, 1e-6))
  expect_true(evaluator_check_close(1e-9, 0, 1e-8, 0))
  expect_false(evaluator_check_close(1e308, -1e308, 0, 1e-6))
})

test_that("point generation preserves RNG and is seeded; explicit points use one batch", {
  # Isolated function environment allows mocking without loading the native DLL.
  env <- new.env(parent = environment(nutpie_validate_density_evaluator))
  validate <- nutpie_validate_density_evaluator
  environment(validate) <- env
  env$evaluator_check_dimension <- function(model) 2L
  captured <- NULL
  env$evaluator_check_evaluate <- function(model, points) {
    captured <<- points
    x <- lapply(seq_len(nrow(points)), function(i) checker_ok(points[i, ]))
    list(reference = x, evaluator = x)
  }
  set.seed(717)
  state <- .Random.seed
  first <- validate(NULL, seed = 42)
  expect_identical(.Random.seed, state)
  expect_identical(first$points, validate(NULL, seed = 42)$points)
  expect_false(identical(first$points, validate(NULL, seed = 43)$points))
  expect_identical(first$status, "pass")
  points <- rbind(c(1, 2), c(3, 4))
  validate(NULL, points = points)
  expect_equal(captured, rbind(points, points[c(1, 2, 1), ]))
  expect_identical(validate(NULL, points = c(1, 2))$status, "inconclusive")
  expect_error(validate(NULL, points = c(1, 2, 3)), "length 2")
  expect_error(validate(NULL, points = c(1, Inf)), "finite")
  expect_error(validate(NULL, num_points = 1), "at least two")
  expect_error(validate(NULL, seed = NA_real_), "seed")
  expect_error(validate(NULL, gradient_rtol = -1), "gradient_rtol")
  expect_error(validate(NULL, radius = 0), "radius")
})


test_that("fatal errors in the repeatability batch cannot become inconclusive", {
  report <- checker_case(function(x) { x[[4]] <- list(status = "fatal", message = "failed"); x })
  expect_identical(report$status, "fail")
  expect_identical(report$repeatability$status, "fail")
})

test_that("native BYOLD fixture matrix passes in bounded child processes", {
  skip_if(Sys.getenv("NUTPIER_RUN_BYOLD_TESTS") != "1",
          "set NUTPIER_RUN_BYOLD_TESTS=1 after a release install")
  fixture_dir <- normalizePath(test_path("byold-v1"), mustWork = TRUE)
  for (mode in 0:16) {
    log <- tempfile(paste0("byold-mode-", mode, "-"), fileext = ".log")
    status <- system2(file.path(R.home("bin"), "Rscript"),
      c(shQuote(file.path(fixture_dir, "native-case.R")), shQuote(fixture_dir), mode),
      stdout = log, stderr = log, timeout = 240)
    output <- paste(readLines(log, warn = FALSE), collapse = "\n")
    expect_equal(status, 0L, info = paste("mode", mode, "log", log, output))
    expect_match(output, paste("BYOLD fixture mode", mode, "passed"))
  }
})
