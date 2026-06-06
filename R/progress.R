#' @noRd
in_with_progress <- function() {
  if (!requireNamespace("progressr", quietly = TRUE)) return(FALSE)
  target <- progressr::with_progress
  for (i in seq_len(sys.nframe())) {
    if (identical(sys.function(i), target)) return(TRUE)
  }
  FALSE
}

#' @noRd
should_use_cli_progress <- function() {
  interactive() && requireNamespace("cli", quietly = TRUE)
}

#' @noRd
resolve_progress_mode <- function(progress, refresh) {
  if (isTRUE(refresh <= 0L) || identical(progress, "none")) return("none")
  switch(
    progress,
    "auto" = if (should_use_cli_progress()) "cli" else "text",
    "cli" = {
      if (!requireNamespace("cli", quietly = TRUE)) {
        stop("`progress = \"cli\"` requires the 'cli' package. ",
             "Install it or set `progress = \"text\"`.", call. = FALSE)
      }
      "cli"
    },
    "progressr" = {
      if (!requireNamespace("progressr", quietly = TRUE)) {
        stop("`progress = \"progressr\"` requires the 'progressr' package. ",
             "Install it or set `progress = \"cli\"`/`\"text\"`.", call. = FALSE)
      }
      "progressr"
    },
    "text" = "text",
    "none" = "none"
  )
}

#' @noRd
as_progress_num <- function(x, default = 0) {
  if (is.null(x) || length(x) == 0L || is.na(x)) return(default)
  as.numeric(x)
}

#' @noRd
format_progress_time <- function(seconds) {
  seconds <- as_progress_num(seconds, 0)
  if (!is.finite(seconds) || seconds < 0) seconds <- 0
  if (seconds < 60) return(sprintf("%.1fs", seconds))
  mins <- floor(seconds / 60)
  secs <- round(seconds %% 60)
  if (mins < 60) return(sprintf("%dm%02ds", mins, secs))
  hours <- floor(mins / 60)
  mins <- mins %% 60
  sprintf("%dh%02dm", hours, mins)
}

#' @noRd
summarize_progress_snapshot <- function(snapshot) {
  if (length(snapshot) == 0L) {
    return(list(
      total_finished = 0, total_draws = 0, phase = "warmup",
      total_divergences = 0L, slowest_chain = NA_integer_,
      min_step_size = NA_real_, max_latest_num_steps = NA_integer_,
      avg_num_steps = NA_real_, first_divergence = NA_character_,
      max_runtime = 0, status = "waiting for chains"
    ))
  }

  finished <- vapply(snapshot, function(s) as_progress_num(s$finished_draws), numeric(1))
  totals <- vapply(snapshot, function(s) as_progress_num(s$total_draws), numeric(1))
  divergences <- vapply(snapshot, function(s) as.integer(as_progress_num(s$divergences)), integer(1))
  tuning <- vapply(snapshot, function(s) isTRUE(s$tuning), logical(1))
  steps <- vapply(snapshot, function(s) as.integer(as_progress_num(s$latest_num_steps)), integer(1))
  total_steps <- vapply(snapshot, function(s) as_progress_num(s$total_num_steps), numeric(1))
  step_sizes <- vapply(snapshot, function(s) as_progress_num(s$step_size, NA_real_), numeric(1))
  runtimes <- vapply(snapshot, function(s) as_progress_num(s$runtime), numeric(1))
  chains <- seq_along(snapshot)
  chain_values <- vapply(snapshot, function(s) as.integer(as_progress_num(s$chain, NA_real_)), integer(1))
  chains[!is.na(chain_values)] <- chain_values[!is.na(chain_values)]

  progress_ratio <- ifelse(totals > 0, finished / totals, 1)
  slowest <- chains[which.min(progress_ratio)]
  finite_steps <- step_sizes[is.finite(step_sizes) & step_sizes > 0]
  min_step <- if (length(finite_steps) > 0L) min(finite_steps) else NA_real_
  avg_steps <- if (sum(finished) > 0) sum(total_steps) / sum(finished) else NA_real_

  first_div <- NA_character_
  div_chains <- which(divergences > 0L)
  if (length(div_chains) > 0L) {
    candidates <- lapply(div_chains, function(i) {
      draws <- snapshot[[i]]$divergent_draws
      if (is.null(draws) || length(draws) == 0L) return(NULL)
      list(chain = chains[i], draw = min(as.integer(draws)))
    })
    candidates <- Filter(Negate(is.null), candidates)
    if (length(candidates) > 0L) {
      draws <- vapply(candidates, `[[`, integer(1), "draw")
      first <- candidates[[which.min(draws)]]
      first_div <- sprintf("chain %d draw %d", first$chain, first$draw)
    }
  }

  phase <- if (all(tuning)) {
    "warmup"
  } else if (any(tuning)) {
    "mixed"
  } else {
    "sample"
  }
  total_divergences <- sum(divergences)
  total_finished <- sum(finished)
  total_draws <- sum(totals)
  max_latest_steps <- if (length(steps) > 0L) max(steps) else NA_integer_
  max_runtime <- if (length(runtimes) > 0L) max(runtimes) else 0

  slowest_idx <- which(chains == slowest)[1]
  diag_bits <- c(
    phase,
    sprintf("c%d %d/%d", slowest, finished[slowest_idx], totals[slowest_idx])
  )
  if (is.finite(min_step)) diag_bits <- c(diag_bits, sprintf("step %.3g", min_step))
  if (is.finite(avg_steps)) diag_bits <- c(diag_bits, sprintf("lf %.1f avg", avg_steps))
  diag_bits <- c(diag_bits, sprintf("lf %d max", max_latest_steps))
  if (total_divergences > 0L) {
    div_msg <- sprintf("div %d", total_divergences)
    if (!is.na(first_div)) div_msg <- paste0(div_msg, " first ", first_div)
    diag_bits <- c(div_msg, diag_bits)
  }

  list(
    total_finished = total_finished,
    total_draws = total_draws,
    phase = phase,
    total_divergences = total_divergences,
    slowest_chain = slowest,
    min_step_size = min_step,
    max_latest_num_steps = max_latest_steps,
    avg_num_steps = avg_steps,
    first_divergence = first_div,
    max_runtime = max_runtime,
    status = paste(diag_bits, collapse = " | ")
  )
}

#' @noRd
print_progress_summary <- function(summary, snapshot, elapsed = NULL) {
  if (is.null(summary) || length(snapshot) == 0L) return(invisible(NULL))
  elapsed <- elapsed %||% summary$max_runtime
  message("Sampling complete in ", format_progress_time(elapsed))

  rows <- lapply(snapshot, function(s) {
    finished <- as_progress_num(s$finished_draws)
    total_steps <- as_progress_num(s$total_num_steps)
    avg_steps <- if (finished > 0) total_steps / finished else NA_real_
    sprintf(
      "  Chain %d: %d draws, %s, step %.3g, avg leapfrog %.1f, %d divergences",
      as.integer(as_progress_num(s$chain, 0)),
      as.integer(finished),
      format_progress_time(as_progress_num(s$runtime)),
      as_progress_num(s$step_size, NA_real_),
      avg_steps,
      as.integer(as_progress_num(s$divergences))
    )
  })
  message(paste(rows, collapse = "\n"))
  if (summary$total_divergences > 0L) {
    message(
      "Warning: ", summary$total_divergences,
      " divergent transition", if (summary$total_divergences == 1L) "" else "s",
      " after warmup. Try increasing `target_accept`, inspecting pairs, or reparameterizing."
    )
  }
  invisible(NULL)
}

#' @noRd
print_sampling_diagnostic_summary <- function(diagnostics, num_chains, elapsed) {
  message("Sampling complete in ", format_progress_time(elapsed))
  if (is.null(diagnostics) || length(diagnostics) == 0L || is.null(diagnostics$chain)) {
    return(invisible(NULL))
  }

  chains <- sort(unique(as.integer(diagnostics$chain)))
  rows <- lapply(chains, function(ch) {
    idx <- as.integer(diagnostics$chain) == ch
    n <- sum(idx)
    step <- if (!is.null(diagnostics$step_size)) {
      tail(stats::na.omit(as.numeric(diagnostics$step_size[idx])), 1)
    } else {
      numeric()
    }
    step <- if (length(step) == 0L) NA_real_ else step[1]
    avg_leapfrog <- if (!is.null(diagnostics$n_steps)) {
      mean(as.numeric(diagnostics$n_steps[idx]), na.rm = TRUE)
    } else {
      NA_real_
    }
    divs <- if (!is.null(diagnostics$diverging)) {
      sum(as.logical(diagnostics$diverging[idx]), na.rm = TRUE)
    } else {
      0L
    }
    sprintf(
      "  Chain %d: %d draws, step %.3g, avg leapfrog %.1f, %d divergences",
      ch, n, step, avg_leapfrog, as.integer(divs)
    )
  })
  message(paste(rows, collapse = "\n"))

  total_divs <- if (!is.null(diagnostics$diverging)) {
    sum(as.logical(diagnostics$diverging), na.rm = TRUE)
  } else {
    0L
  }
  if (total_divs > 0L) {
    message(
      "Warning: ", total_divs,
      " divergent transition", if (total_divs == 1L) "" else "s",
      " after warmup. Try increasing `target_accept`, inspecting pairs, or reparameterizing."
    )
  }
  invisible(NULL)
}

#' @noRd
make_cli_progress_callback <- function(num_chains, num_warmup, num_draws,
                                       id = NULL,
                                       update = NULL,
                                       done = NULL) {
  if (is.null(update)) update <- cli::cli_progress_update
  if (is.null(done)) done <- cli::cli_progress_done
  total_steps <- as.numeric(num_chains) * (as.numeric(num_warmup) + as.numeric(num_draws))
  if (is.null(id)) {
    id <- cli::cli_progress_bar(
      name = "Sampling",
      total = total_steps,
      type = "custom",
      clear = FALSE,
      format = paste(
        "{cli::pb_spin} Sampling {cli::pb_percent} |{cli::pb_bar}|",
        "{cli::pb_current}/{cli::pb_total} draws {cli::pb_eta_str}",
        "{cli::pb_status}"
      ),
      format_done = paste(
        "Sampling {cli::pb_percent} |{cli::pb_bar}|",
        "{cli::pb_current}/{cli::pb_total} draws",
        "{cli::pb_status}"
      ),
      .auto_close = FALSE
    )
  }

  last_total <- 0
  last_status <- ""
  last_summary <- NULL
  last_snapshot <- NULL

  callback_failed <- FALSE

  callback <- function(snapshot) {
    if (callback_failed) return(invisible(NULL))
    summary <- summarize_progress_snapshot(snapshot)
    last_summary <<- summary
    last_snapshot <<- snapshot
    total_now <- min(summary$total_finished, total_steps)
    status <- summary$status
    if (total_now == last_total && identical(status, last_status)) return(invisible(NULL))
    last_total <<- total_now
    last_status <<- status
    ok <- try(update(set = total_now, status = status, id = id, force = TRUE), silent = TRUE)
    if (inherits(ok, "try-error")) callback_failed <<- TRUE
    invisible(NULL)
  }

  attr(callback, "finish") <- function() {
    try(done(id = id), silent = TRUE)
    invisible(NULL)
  }
  callback
}

#' @noRd
make_progressr_callback <- function(num_chains, num_warmup, num_draws) {
  total_steps <- as.numeric(num_chains) * (as.numeric(num_warmup) + as.numeric(num_draws))
  p <- progressr::progressor(steps = total_steps, on_exit = FALSE)
  last_total <- 0
  last_status <- ""
  function(snapshot) {
    summary <- summarize_progress_snapshot(snapshot)
    delta <- max(0, min(summary$total_finished, total_steps) - last_total)
    last_total <<- min(summary$total_finished, total_steps)
    if (delta == 0 && identical(summary$status, last_status)) return(invisible(NULL))
    last_status <<- summary$status
    p(amount = delta, message = summary$status)
    invisible(NULL)
  }
}

#' @noRd
finish_progress_callback <- function(callback) {
  finish <- attr(callback, "finish")
  if (is.function(finish)) finish()
  invisible(NULL)
}
