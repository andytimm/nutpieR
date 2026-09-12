#' Compare a density evaluator with its BridgeStan reference
#'
#' Compare log density and the full gradient at broad random unconstrained
#' points by default. Use `method = "reference"` to add positions from a short
#' BridgeStan-only pilot. The target is `propto = TRUE, jacobian = TRUE`.
#' Possible constant offsets are reported, not corrected; ordinary tolerances
#' still apply.
#'
#' This advisory check runs trusted native code inside R. It cannot certify
#' memory safety or agreement outside the tested points. Sampling does not run
#' this check automatically.
#'
#' @param model A bound model from [nutpie_attach_density_evaluator()].
#' @param points Advanced: numeric matrix with one unconstrained point per row, or a
#'   numeric vector for one point. These are not constrained parameter values.
#' @param num_points Number of random points (default method) or retained
#'   reference-pilot positions when `points` is `NULL` (at least two).
#' @param seed Seed for generated points. The caller's RNG state is preserved.
#' @param radius Broad random coordinates are uniform on `[-radius, radius]`.
#' @param method `"random"` (default) checks `num_points` broad random points.
#'   `"reference"` checks `num_points` pilot positions plus four random points.
#'   Cannot be combined with explicit `points`.
#' @section Reference pilot:
#' The pilot reuses the bound reference and its data realization. It uses one
#' chain/core, 200 warmup iterations, `num_points` retained draws, diagonal
#' adaptation, target acceptance 0.8, maximum tree depth 10, and uniform
#' `[-2, 2]` initialization. It never evaluates the evaluator and does not expand
#' constrained outputs or generated quantities.
#'
#' This adds sampling cost; elapsed time and retained-draw diagnostics are in
#' `$pilot`. Failure stops the check, with no random fallback. A short pilot
#' does not guarantee convergence or typical-set coverage.
#'
#' @section Guide:
#' See `system.file("examples/byold/README.md", package = "nutpieR")` for the
#' evaluator-writing workflow, layout rules, and interpretation of this check.
#' @param logp_atol,logp_rtol Absolute and relative log-density tolerances.
#' @param gradient_atol,gradient_rtol Absolute and relative gradient tolerances.
#' @return A `nutpie_density_evaluator_check` list containing an overall `status` (`pass`,
#'   `fail`, or `inconclusive`), point and coordinate comparisons, repeatability,
#'   tolerances, and untested checks. `method`, `point_source`, and `groups`
#'   distinguish explicit, reference-pilot, and random points. Group statuses
#'   are pointwise; repeatability is reported separately. `pilot` records
#'   settings, elapsed seconds, and retained-draw diagnostics (or is `NULL`).
#'   Invalid reference points cannot establish
#'   agreement. Repeatability requires at least two distinct points: the evaluator
#'   evaluates q1, q2, then q1 again using one workspace on the same thread.
#' @export
nutpie_validate_density_evaluator <- function(model, points = NULL, num_points = 10L,
                                   seed = 1L, radius = 2,
                                   logp_atol = 1e-8, logp_rtol = 1e-6,
                                   gradient_atol = 1e-8, gradient_rtol = 1e-6,
                                   method = c("random", "reference")) {
  method <- match.arg(method)
  if (!is.null(points) && method == "reference")
    stop('points cannot be combined with method = "reference".', call. = FALSE)
  pilot <- NULL
  tolerances <- list(logp_atol = logp_atol, logp_rtol = logp_rtol,
                     gradient_atol = gradient_atol, gradient_rtol = gradient_rtol)
  for (name in names(tolerances)) {
    value <- tolerances[[name]]
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value < 0)
      stop(name, " must be one finite nonnegative number.", call. = FALSE)
  }
  dim <- evaluator_check_dimension(model)
  generated <- is.null(points)
  if (generated) {
    if (!is.numeric(num_points) || length(num_points) != 1L ||
        !is.finite(num_points) || num_points < 2 || num_points > .Machine$integer.max ||
        num_points != floor(num_points))
      stop("num_points must be an integer of at least two.", call. = FALSE)
    if (!is.numeric(radius) || length(radius) != 1L || !is.finite(radius) || radius <= 0)
      stop("radius must be one finite positive number.", call. = FALSE)
    if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed) ||
        seed < 0 || seed > .Machine$integer.max || seed != floor(seed))
      stop("seed must be a nonnegative R integer.", call. = FALSE)
    had_rng <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    if (had_rng) old_rng <- get(".Random.seed", envir = .GlobalEnv)
    on.exit({
      if (had_rng) assign(".Random.seed", old_rng, envir = .GlobalEnv)
      else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
        rm(".Random.seed", envir = .GlobalEnv)
    }, add = TRUE)
    set.seed(as.integer(seed))
    if (method == "reference") {
      result <- evaluator_check_pilot(model, as.integer(num_points), as.integer(seed))
      points <- result$points
      if (!is.matrix(points) || !is.numeric(points) ||
          !identical(base::dim(points), c(as.integer(num_points), dim)) ||
          any(!is.finite(points)))
        stop("Reference pilot returned incomplete or invalid positions; check discarded.", call. = FALSE)
      pilot <- result$pilot
      points <- rbind(points, matrix(stats::runif(4L * dim, -radius, radius), ncol = dim))
      point_source <- c(rep("reference", num_points), rep("random", 4L))
    } else {
      points <- matrix(stats::runif(num_points * dim, -radius, radius), ncol = dim)
      point_source <- rep("random", num_points)
    }
  } else {
    if (is.numeric(points) && is.null(base::dim(points))) points <- matrix(points, nrow = 1L)
    if (!is.matrix(points) || !is.numeric(points) || ncol(points) != dim ||
        nrow(points) < 1L || any(!is.finite(points)))
      stop("points must contain finite unconstrained rows of length ", dim, ".", call. = FALSE)
  }
  if (!generated) {
    method <- "explicit"
    point_source <- rep("explicit", nrow(points))
  }
  # Keep q1, q2, q1 in one native batch so they reuse the same-thread workspace.
  other <- which(vapply(seq_len(nrow(points)), function(i)
    any(points[i, ] != points[1L, ]), logical(1)))
  repeat_rows <- if (length(other)) c(1L, other[1L], 1L) else integer()
  evaluation_points <- rbind(points, points[repeat_rows, , drop = FALSE])
  values <- evaluator_check_evaluate(model, evaluation_points)
  report <- evaluator_check_report(points, values, tolerances, repeat_rows,
                                if (generated) seed else NULL)
  report$method <- method
  report$point_source <- point_source
  report$comparisons$point_source <- point_source
  if (!is.null(report$gradients))
    report$gradients$point_source <- point_source[report$gradients$point]
  report$groups <- do.call(rbind, lapply(unique(point_source), function(source) {
    statuses <- report$comparisons$status[point_source == source]
    data.frame(point_source = source, n = length(statuses),
      status = if (any(statuses == "fail")) "fail" else
        if (all(statuses == "pass")) "pass" else "inconclusive",
      pass = sum(statuses == "pass"), fail = sum(statuses == "fail"),
      inconclusive = sum(statuses == "inconclusive"))
  }))
  report$pilot <- pilot
  report
}

evaluator_check_pilot <- function(model, num_points, seed) {
  # Check the reference owner before paying for the pilot; never evaluate the evaluator.
  bs_ndim_unc(model$bs_ptr)
  start <- proc.time()[["elapsed"]]
  result <- tryCatch(bs_reference_pilot(model$bs_ptr, num_points, seed),
    error = function(e) stop("Reference pilot failed; check discarded: ",
                             conditionMessage(e), call. = FALSE))
  list(points = result$points, pilot = list(
    settings = list(num_chains = 1L, cores = 1L, num_warmup = 200L,
      num_draws = num_points, seed = seed, target_accept = 0.8,
      max_treedepth = 10L, adaptation = "diag", init_radius = 2),
    elapsed_seconds = unname(proc.time()[["elapsed"]] - start),
    diagnostics = result$diagnostics, sampler_config = result$sampler_config))
}

# Keep native calls separate so report logic can be tested without a library.
evaluator_check_dimension <- function(model) {
  if (!inherits(model, "nutpie_density_evaluator_model"))
    stop("model must be a bound model from nutpie_attach_density_evaluator().", call. = FALSE)
  dim <- model$ndim
  if (!is.numeric(dim) || length(dim) != 1L || !is.finite(dim) || dim < 1 || dim != floor(dim))
    stop("Bound model has an invalid dimension; reattach the evaluator.", call. = FALSE)
  as.integer(dim)
}
evaluator_check_evaluate <- function(model, points) {
  point_list <- lapply(seq_len(nrow(points)), function(i) as.numeric(points[i, ]))
  normalize <- function(values) lapply(values, function(value) {
    value$status <- switch(as.character(value$status),
      `0` = "ok", `1` = "domain", `2` = "fatal", "fatal")
    value
  })
  list(reference = normalize(bs_evaluate(model$bs_ptr, point_list)),
       evaluator = normalize(density_evaluator_evaluate(model$evaluator_ptr, point_list)))
}

evaluator_check_value <- function(value, dim) {
  if (!is.list(value) || !identical(value$status, "ok")) return(FALSE)
  is.numeric(value$logp) && length(value$logp) == 1L && is.finite(value$logp) &&
    is.numeric(value$gradient) && length(value$gradient) == dim &&
    all(is.finite(value$gradient))
}

evaluator_check_close <- function(actual, reference, atol, rtol) {
  # Scale before subtraction to avoid overflow for opposite large finite values.
  scale <- pmax(1, abs(actual), abs(reference))
  abs(actual / scale - reference / scale) <= atol / scale + rtol * abs(reference / scale)
}

evaluator_check_report <- function(points, values, tolerances, repeat_rows, seed) {
  n <- nrow(points)
  dim <- ncol(points)
  expected <- n + length(repeat_rows)
  if (!is.list(values) || length(values$reference) != expected ||
      length(values$evaluator) != expected)
    stop("Native checker returned an invalid batch length.", call. = FALSE)
  t <- tolerances
  compare <- function(a, b) {
    c(evaluator_check_close(a$logp, b$logp, t$logp_atol, t$logp_rtol),
      evaluator_check_close(a$gradient, b$gradient, t$gradient_atol, t$gradient_rtol))
  }
  rows <- vector("list", n)
  coordinates <- vector("list", n)
  offsets <- numeric()
  gradients_agree <- TRUE
  for (i in seq_len(n)) {
    r <- values$reference[[i]]
    k <- values$evaluator[[i]]
    rv <- evaluator_check_value(r, dim)
    kv <- evaluator_check_value(k, dim)
    status <- "inconclusive"
    reason <- "reference_invalid"
    lp_error <- NA_real_
    lp_ok <- NA
    grad_ok <- NA
    if (rv) {
      if (!kv) {
        status <- "fail"
        reason <- "evaluator_invalid"
      } else {
        agreement <- compare(k, r)
        lp_ok <- agreement[1L]
        grad_ok <- all(agreement[-1L])
        status <- if (all(agreement)) "pass" else "fail"
        reason <- if (all(agreement)) "agreement" else "numerical_mismatch"
        lp_error <- k$logp - r$logp
        offsets <- c(offsets, lp_error)
        gradients_agree <- gradients_agree && grad_ok
        coordinates[[i]] <- data.frame(point = i, coordinate = seq_len(dim),
          reference = r$gradient, evaluator = k$gradient,
          error = k$gradient - r$gradient, pass = agreement[-1L])
      }
    } else if (is.list(k) && (identical(k$status, "fatal") ||
               (identical(k$status, "ok") && !kv))) {
      status <- "fail"
      reason <- "evaluator_invalid"
    }
    outcome <- function(v) if (is.list(v) && length(v$status) == 1L) as.character(v$status) else "malformed"
    message <- function(v) if (is.list(v) && length(v$message) == 1L) as.character(v$message) else ""
    rows[[i]] <- data.frame(point = i, status = status, reason = reason,
      reference_status = outcome(r), evaluator_status = outcome(k),
      reference_message = message(r), evaluator_message = message(k),
      logp_error = lp_error, logp_pass = lp_ok, gradient_pass = grad_ok)
  }
  comparisons <- do.call(rbind, rows)
  repeatability <- list(status = "inconclusive", reason = "need_two_distinct_points")
  if (length(repeat_rows)) {
    a <- values$evaluator[[n + 1L]]
    b <- values$evaluator[[n + 3L]]
    valid <- evaluator_check_value(a, dim) && evaluator_check_value(b, dim)
    repeat_values <- values$evaluator[n + seq_len(3L)]
    broken <- any(vapply(repeat_values, function(v)
      !is.list(v) || identical(v$status, "fatal") ||
        (identical(v$status, "ok") && !evaluator_check_value(v, dim)), logical(1)))
    repeatability <- list(status = if (broken) "fail" else if (!valid) "inconclusive" else
      if (all(compare(a, b))) "pass" else "fail",
      reason = if (!valid) "invalid_repeated_point" else "interleaved_q1_q2_q1",
      point_indices = repeat_rows,
      first = a, repeated = b)
  }
  constant_offset <- list(status = "untested", offset = NA_real_)
  if (length(offsets) >= 2L && all(is.finite(offsets)) && gradients_agree) {
    stable <- all(evaluator_check_close(offsets, offsets[1L], t$logp_atol, t$logp_rtol))
    nonzero <- any(!evaluator_check_close(offsets, 0, t$logp_atol, 0))
    constant_offset <- list(status = if (stable && nonzero) "possible_constant_offset" else "not_detected",
                            offset = if (stable) offsets[1L] else NA_real_)
  }
  status <- if (any(comparisons$status == "fail") || repeatability$status == "fail") "fail" else
    if (all(comparisons$status == "pass") && repeatability$status == "pass") "pass" else "inconclusive"
  structure(list(status = status, advisory = TRUE,
    convention = list(propto = TRUE, jacobian = TRUE), points = points, seed = seed,
    tolerances = tolerances, comparisons = comparisons,
    gradients = do.call(rbind, coordinates), repeatability = repeatability,
    constant_offset = constant_offset,
    counts = table(factor(comparisons$status, levels = c("pass", "fail", "inconclusive"))),
    untested = c("concurrent_workspaces", "memory_safety", "all_parameter_space")),
    class = "nutpie_density_evaluator_check")
}
