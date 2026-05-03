# Progress reporting for nutpie_sample().
#
# The Rust-side poll loop pushes a per-chain snapshot (a list of
# `num_chains` named lists with `finished_draws`, `total_draws`,
# `divergences`, `tuning`, `step_size`) on every wakeup. R-side handlers
# render whatever they want from that snapshot — we ship a progressr-driven
# default and a "text" regression fallback.

#' @noRd
should_use_progressr <- function() {
  interactive() &&
    requireNamespace("progressr", quietly = TRUE) &&
    requireNamespace("cli", quietly = TRUE)
}

# Build a closure suitable for passing as `progress_callback` in `sample_stan`.
# Reports total finished draws (warmup + sample) against the full budget so a
# single combined bar tracks the run end-to-end. Per-chain bars introduce ANSI
# rendering issues in RStudio (cmdstanr hit the same wall in PR #1138).
#' @noRd
make_progressr_callback <- function(num_chains, num_warmup, num_draws) {
  total_steps <- as.numeric(num_chains) * (as.numeric(num_warmup) + as.numeric(num_draws))
  p <- progressr::progressor(steps = total_steps)
  last_total <- 0
  function(snapshot) {
    finished <- vapply(snapshot, function(s) as.numeric(s$finished_draws %||% 0),
                       numeric(1))
    divergences <- sum(vapply(snapshot,
                              function(s) as.integer(s$divergences %||% 0L),
                              integer(1)))
    any_tuning <- any(vapply(snapshot, function(s) isTRUE(s$tuning), logical(1)))
    step_sizes <- vapply(snapshot,
                         function(s) as.numeric(s$step_size %||% NA_real_),
                         numeric(1))

    total_now <- sum(finished)
    delta <- max(0, total_now - last_total)
    last_total <<- total_now

    min_step <- suppressWarnings(min(step_sizes, na.rm = TRUE))
    step_msg <- if (is.finite(min_step)) sprintf(", step %.3f", min_step) else ""
    div_msg <- if (divergences > 0) sprintf(", %d divergences", divergences) else ""
    label <- sprintf("nutpie [%s] %d/%d draws%s%s",
                     if (any_tuning) "warmup" else "sample",
                     as.integer(total_now), as.integer(total_steps),
                     div_msg, step_msg)

    p(amount = delta, message = label)
    invisible(NULL)
  }
}

# If no progressr handler is registered for the session, register a cli-styled
# one so users see a bar without having to wrap their call in
# `progressr::with_progress({...})`. Mirrors cmdstanr PR #1138's
# `register_default_progressr_handler()` helper.
#' @noRd
register_default_progress_handler <- function() {
  if (!requireNamespace("progressr", quietly = TRUE)) return(invisible(FALSE))
  if (length(progressr::handlers()) > 0L) return(invisible(FALSE))
  handler <- if (requireNamespace("cli", quietly = TRUE)) {
    progressr::handler_cli()
  } else {
    "txtprogressbar"
  }
  progressr::handlers(global = TRUE, handler)
  invisible(TRUE)
}
