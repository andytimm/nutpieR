#' @noRd
should_use_progressr <- function() {
  interactive() &&
    requireNamespace("progressr", quietly = TRUE) &&
    requireNamespace("cli", quietly = TRUE)
}

# Combined-bar progressr callback. Per-chain bars hit ANSI rendering issues in
# RStudio (cmdstanr ran into the same in PR #1138).
#' @noRd
make_progressr_callback <- function(num_chains, num_warmup, num_draws) {
  total_steps <- as.numeric(num_chains) * (as.numeric(num_warmup) + as.numeric(num_draws))
  p <- progressr::progressor(steps = total_steps)
  last_total <- 0
  last_label <- ""
  function(snapshot) {
    total_now <- 0
    divergences <- 0L
    any_tuning <- FALSE
    min_step <- Inf
    for (s in snapshot) {
      total_now <- total_now + as.numeric(s$finished_draws %||% 0)
      divergences <- divergences + as.integer(s$divergences %||% 0L)
      if (isTRUE(s$tuning)) any_tuning <- TRUE
      step <- as.numeric(s$step_size %||% NA_real_)
      if (is.finite(step) && step < min_step) min_step <- step
    }
    delta <- max(0, total_now - last_total)
    last_total <<- total_now

    step_msg <- if (is.finite(min_step)) sprintf(", step %.3f", min_step) else ""
    div_msg <- if (divergences > 0) sprintf(", %d divergences", divergences) else ""
    label <- sprintf("nutpie [%s] %d/%d draws%s%s",
                     if (any_tuning) "warmup" else "sample",
                     as.integer(total_now), as.integer(total_steps),
                     div_msg, step_msg)

    if (delta == 0 && identical(label, last_label)) return(invisible(NULL))
    last_label <<- label
    p(amount = delta, message = label)
    invisible(NULL)
  }
}

#' @noRd
register_default_progress_handler <- function() {
  if (!requireNamespace("progressr", quietly = TRUE)) return(invisible(FALSE))
  # progressr::handlers() always returns the package-level txtprogressbar
  # default in a fresh session, so length() > 0 is meaningless. The user's
  # explicit configuration lives in `options(progressr.handlers = ...)`;
  # NULL means they haven't customized.
  if (!is.null(getOption("progressr.handlers"))) return(invisible(FALSE))
  handler <- if (requireNamespace("cli", quietly = TRUE)) {
    progressr::handler_cli()
  } else {
    progressr::handler_txtprogressbar()
  }
  progressr::handlers(handler)
  # Enabling global rendering fails when calling handlers are already on the
  # stack — which is exactly what `progressr::with_progress({...})` does. In
  # that case the user has already opted into rendering, so we just leave the
  # handler list set and let `with_progress` drive it.
  tryCatch(
    progressr::handlers(global = TRUE),
    error = function(e) NULL
  )
  invisible(TRUE)
}
