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
format_draw_count <- function(x) {
  format(as.integer(x), big.mark = ",", trim = TRUE, scientific = FALSE)
}

#' @noRd
format_divergence_status <- function(total_divs) {
  if (total_divs == 0L) return("div: 0")
  warn_sym <- if (requireNamespace("cli", quietly = TRUE)) cli::symbol$warning else "!"
  sprintf("%s div: %d", warn_sym, as.integer(total_divs))
}

#' @noRd
infer_tree_depth <- function(n_steps) {
  if (!is.finite(n_steps) || n_steps <= 0) return(NA_integer_)
  as.integer(ceiling(log2(n_steps + 1)))
}

#' @noRd
style_progress_status <- function(status, color = FALSE) {
  if (!isTRUE(color) || !requireNamespace("cli", quietly = TRUE)) return(status)
  color_match <- function(s, pattern, fn) {
    m <- regexpr(pattern, s)
    if (m[[1]] > 0L) regmatches(s, m) <- fn(regmatches(s, m))
    s
  }
  status <- color_match(status, "[^ ]+ div: [1-9][0-9]*", cli::col_red)
  color_match(status, "~ [0-9.]+ grad/draw", cli::col_yellow)
}

#' @noRd
format_gradient_status <- function(avg_lf, max_treedepth = 10L) {
  if (!is.finite(avg_lf)) return("- grad/draw")
  max_possible <- 2^as.integer(max_treedepth %||% 10L) - 1L
  label <- sprintf("%.1f grad/draw", avg_lf)
  if (avg_lf / max_possible >= 0.05) paste("~", label) else label
}

#' @noRd
summarize_progress_snapshot <- function(snapshot, max_treedepth = 10L) {
  if (length(snapshot) == 0L) {
    return(list(
      total_finished = 0, total_draws = 0, phase = "warmup",
      phase_label = "warmup",
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

  num_tuning <- sum(tuning)
  phase <- if (all(tuning)) {
    "warmup"
  } else if (any(tuning)) {
    "mixed"
  } else {
    "sample"
  }
  phase_label <- if (all(tuning)) {
    "warmup"
  } else if (any(tuning)) {
    sprintf("warmup %d/%d", num_tuning, length(tuning))
  } else {
    "sample"
  }
  total_divergences <- sum(divergences)
  total_finished <- sum(finished)
  total_draws <- sum(totals)
  max_latest_steps <- if (length(steps) > 0L) max(steps) else NA_integer_
  max_runtime <- if (length(runtimes) > 0L) max(runtimes) else 0
  status <- paste(
    format_divergence_status(sum(divergences, na.rm = TRUE)),
    format_gradient_status(avg_steps, max_treedepth),
    sep = " | "
  )

  list(
    total_finished = total_finished,
    total_draws = total_draws,
    phase = phase,
    phase_label = phase_label,
    total_divergences = total_divergences,
    slowest_chain = NA_integer_,
    min_step_size = min_step,
    max_latest_num_steps = max_latest_steps,
    avg_num_steps = avg_steps,
    first_divergence = first_div,
    max_runtime = max_runtime,
    status = status
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
format_progress_value <- function(x, digits = NULL) {
  if (!is.finite(x)) return("NA")
  if (is.null(digits)) return(format_draw_count(x))
  sprintf(paste0("%.", digits, "f"), x)
}

#' @noRd
progress_supports_color <- function() {
  requireNamespace("cli", quietly = TRUE) && cli::num_ansi_colors() > 1L
}

#' @noRd
render_progress_table <- function(table) {
  headers <- names(table)
  values <- lapply(table, as.character)
  widths <- vapply(seq_along(headers), function(i) {
    max(nchar(c(headers[i], values[[i]]), type = "width"), na.rm = TRUE)
  }, integer(1))
  align_right <- headers != "chain"
  format_row <- function(row) {
    cells <- vapply(seq_along(headers), function(i) {
      value <- as.character(row[[i]])
      if (align_right[i]) sprintf(paste0("%", widths[i], "s"), value)
      else sprintf(paste0("%-", widths[i], "s"), value)
    }, character(1))
    paste(cells, collapse = "  ")
  }
  header <- format_row(stats::setNames(as.list(headers), headers))
  sep <- paste(vapply(widths, function(w) paste(rep("-", w), collapse = ""), character(1)), collapse = "  ")
  body <- vapply(seq_len(nrow(table)), function(i) format_row(table[i, , drop = FALSE]), character(1))
  c(header, sep, body)
}

#' @noRd
print_sampling_diagnostic_summary <- function(diagnostics, num_chains, elapsed) {
  elapsed_label <- format_progress_time(elapsed)
  if (is.null(diagnostics) || length(diagnostics) == 0L || is.null(diagnostics$chain)) {
    message("Sampling complete in ", elapsed_label)
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
    avg_grad <- if (!is.null(diagnostics$n_steps)) {
      mean(as.numeric(diagnostics$n_steps[idx]), na.rm = TRUE)
    } else {
      NA_real_
    }
    max_tdepth <- if (!is.null(diagnostics$depth)) {
      max(as.numeric(diagnostics$depth[idx]), na.rm = TRUE)
    } else if (!is.null(diagnostics$n_steps)) {
      infer_tree_depth(max(as.numeric(diagnostics$n_steps[idx]), na.rm = TRUE))
    } else {
      NA_real_
    }
    divs <- if (!is.null(diagnostics$diverging)) {
      sum(as.logical(diagnostics$diverging[idx]), na.rm = TRUE)
    } else {
      0L
    }
    data.frame(
      chain = paste0("c", ch),
      draws = format_draw_count(n),
      `grad/draw` = format_progress_value(avg_grad, digits = 1),
      tdepth = format_progress_value(max_tdepth),
      step = if (is.finite(step)) sprintf("%.3g", step) else "NA",
      div = as.integer(divs),
      check.names = FALSE
    )
  })
  table <- do.call(rbind, rows)
  total_divs <- sum(table$div, na.rm = TRUE)

  if (requireNamespace("cli", quietly = TRUE)) {
    if (total_divs > 0L) {
      cli::cli_alert_warning("Sampling complete in {elapsed_label} with {total_divs} divergent transition{?s}.")
    } else {
      cli::cli_alert_success("Sampling complete in {elapsed_label} with no divergences.")
    }
    display <- table
    if (progress_supports_color() && any(display$div > 0L)) {
      display$div <- ifelse(display$div > 0L, cli::col_red(display$div), display$div)
    }
    message(paste(render_progress_table(display), collapse = "\n"))
  } else {
    message(
      "Sampling complete in ", elapsed_label,
      if (total_divs > 0L) paste0(" with ", total_divs, " divergences.") else " with no divergences."
    )
    rows <- apply(table, 1, function(row) {
      sprintf(
        "  %s: %s draws, grad/draw %s, tdepth %s, step %s, div %s",
        row[["chain"]], row[["draws"]], row[["grad/draw"]],
        row[["tdepth"]], row[["step"]], row[["div"]]
      )
    })
    message(paste(rows, collapse = "\n"))
  }

  if (total_divs > 0L) {
    msg <- paste0(
      "Try increasing `target_accept`, inspecting pairs, or reparameterizing."
    )
    if (requireNamespace("cli", quietly = TRUE)) cli::cli_alert_info(msg) else message(msg)
  }
  invisible(NULL)
}

#' @noRd
make_cli_progress_callback <- function(num_chains, num_warmup, num_draws,
                                       max_treedepth = 10L,
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
      extra = list(phase = "warmup"),
      format = paste(
        "{cli::pb_spin} {cli::pb_extra$phase} {cli::pb_percent} |{cli::pb_bar}|",
        "{cli::pb_current}/{cli::pb_total} {cli::pb_eta_str}",
        "| {cli::pb_status}"
      ),
      format_done = paste(
        "{cli::pb_percent} |{cli::pb_bar}|",
        "{cli::pb_current}/{cli::pb_total}",
        "| {cli::pb_status}"
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
    summary <- summarize_progress_snapshot(snapshot, max_treedepth = max_treedepth)
    last_summary <<- summary
    last_snapshot <<- snapshot
    total_now <- min(summary$total_finished, total_steps)
    status <- style_progress_status(summary$status, color = progress_supports_color())
    if (total_now == last_total && identical(status, last_status)) return(invisible(NULL))
    last_total <<- total_now
    last_status <<- status
    ok <- try(update(
      set = total_now,
      status = status,
      extra = list(phase = summary$phase_label),
      id = id,
      force = TRUE
    ), silent = TRUE)
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
make_progressr_callback <- function(num_chains, num_warmup, num_draws,
                                    max_treedepth = 10L) {
  total_steps <- as.numeric(num_chains) * (as.numeric(num_warmup) + as.numeric(num_draws))
  p <- progressr::progressor(steps = total_steps, on_exit = FALSE)
  last_total <- 0
  last_status <- ""
  callback_failed <- FALSE
  function(snapshot) {
    if (callback_failed) return(invisible(NULL))
    summary <- summarize_progress_snapshot(snapshot, max_treedepth = max_treedepth)
    delta <- max(0, min(summary$total_finished, total_steps) - last_total)
    last_total <<- min(summary$total_finished, total_steps)
    full_status <- paste0(summary$phase_label, " ", summary$status)
    if (delta == 0 && identical(full_status, last_status)) return(invisible(NULL))
    last_status <<- full_status
    ok <- try(p(amount = delta, message = full_status), silent = TRUE)
    if (inherits(ok, "try-error")) callback_failed <<- TRUE
    invisible(NULL)
  }
}

#' @noRd
finish_progress_callback <- function(callback) {
  finish <- attr(callback, "finish")
  if (is.function(finish)) finish()
  invisible(NULL)
}
