kernel_print_fixture <- function(change = identity, reference_change = identity,
                                 points = rbind(c(1, 2), c(2, 3)), repeat_rows = c(1L, 2L, 1L)) {
  order <- c(seq_len(nrow(points)), repeat_rows)
  values <- lapply(order, function(i) list(status = "ok", message = "",
    logp = -sum(points[i, ]^2) / 2, gradient = -points[i, ]))
  kernel_check_report(points,
    list(reference = reference_change(values), kernel = change(values)),
    list(logp_atol = 1e-8, logp_rtol = 1e-6,
         gradient_atol = 1e-8, gradient_rtol = 1e-6), repeat_rows, NULL)
}

test_that("passing checks print a short summary without changing the report", {
  x <- kernel_print_fixture()
  before <- serialize(x, NULL)
  output <- capture.output(result <- withVisible(print.nutpie_density_kernel_check(x)))
  expect_false(result$visible)
  expect_identical(result$value, x)
  expect_identical(serialize(x, NULL), before)
  expect_length(output, 4L)
  expect_match(output[1L], "pass (advisory); points: 2 pass, 0 fail, 0 inconclusive", fixed = TRUE)
  expect_match(output[2L], "logp 0; gradient 0", fixed = TRUE)
  expect_match(output[3L], "Repeatability: pass", fixed = TRUE)
  expect_match(output[4L], "$comparisons", fixed = TRUE)
})

test_that("failed checks show the first point and gradient coordinate", {
  x <- kernel_print_fixture(function(v) { v[[2L]]$gradient[2L] <- 0; v })
  output <- capture.output(print.nutpie_density_kernel_check(x))
  expect_match(output[1L], "Density kernel check: fail", fixed = TRUE)
  expect_match(output[4L], "First failure: point 2, coordinate 2", fixed = TRUE)
})

test_that("inconclusive checks explain missing repeatability and differences", {
  reject <- function(v) lapply(v, function(z) list(status = "domain", message = "outside domain"))
  x <- kernel_print_fixture(reference_change = reject,
                            points = matrix(c(1, 2), nrow = 1L), repeat_rows = integer())
  output <- capture.output(print.nutpie_density_kernel_check(x))
  expect_match(output[1L], "Density kernel check: inconclusive", fixed = TRUE)
  expect_match(output[2L], "logp not available; gradient not available", fixed = TRUE)
  expect_match(output[3L], "need two distinct points", fixed = TRUE)
  x <- kernel_print_fixture(change = reject, reference_change = reject)
  expect_match(paste(capture.output(print.nutpie_density_kernel_check(x)), collapse = "\n"),
               "invalid repeated point", fixed = TRUE)
})

test_that("offset notes preserve ordinary tolerance decisions", {
  offset <- function(amount) function(v) lapply(v, function(z) { z$logp <- z$logp + amount; z })
  x <- kernel_print_fixture(change = offset(1e-7))
  expect_identical(x$status, "pass")
  output <- capture.output(print.nutpie_density_kernel_check(x))
  expect_match(output[4L], "within ordinary logp tolerances; not corrected", fixed = TRUE)
  x <- kernel_print_fixture(change = offset(10))
  expect_identical(x$status, "fail")
  output <- capture.output(print.nutpie_density_kernel_check(x))
  expect_match(output[5L], "no tolerance exemption; not corrected", fixed = TRUE)
})

test_that("native error messages cannot flood the console", {
  x <- kernel_print_fixture(function(v) {
    v[[1L]] <- list(status = "fatal", message = paste(rep("failure\n", 1000), collapse = ""))
    v
  })
  before <- serialize(x, NULL)
  output <- capture.output(print.nutpie_density_kernel_check(x))
  expect_length(output, 5L)
  expect_true(all(nchar(output) < 230L))
  expect_match(output[4L], "failure failure", fixed = TRUE)
  expect_match(output[4L], "...", fixed = TRUE)
  expect_identical(serialize(x, NULL), before)
})

test_that("repeatability-only failures and overflow remain visible", {
  x <- kernel_print_fixture(function(v) { v[[5L]]$gradient[1L] <- 100; v })
  output <- capture.output(print.nutpie_density_kernel_check(x))
  expect_match(output[1L], "Density kernel check: fail", fixed = TRUE)
  expect_match(output[3L], "Repeatability: fail", fixed = TRUE)
  x$comparisons$logp_error[1L] <- Inf
  output <- capture.output(print.nutpie_density_kernel_check(x))
  expect_match(output[2L], "logp Inf", fixed = TRUE)
})

test_that("bound density kernel models expose and print their public identity", {
  model <- list2env(list(kernel_path = "/tmp/kernel.so", ndim = 2L),
                    parent = emptyenv())
  class(model) <- c("nutpie_density_kernel_model", "nutpie_model")
  output <- capture.output(result <- withVisible(print.nutpie_density_kernel_model(model)))
  expect_false(result$visible)
  expect_identical(result$value, model)
  expect_match(output[1L], "Experimental density kernel binding", fixed = TRUE)
  expect_match(output[3L], "2 unconstrained parameters", fixed = TRUE)
})
