# Tunable thresholds for the live progress hints and end-of-run summary.
# Collected here so they are easy to find and adjust during dogfooding.
LATE_WARMUP_FRACTION  <- 0.75  # baseline grad/draw stats from this point on
GRAD_HINT_THRESHOLD   <- 128   # avg grad/draw at/above this triggers the hint + accent
SPREAD_TRIGGER_POINTS <- 15    # percentage-point chain spread to trigger the hint
SPREAD_MEDIAN_FLOOR   <- 0.10  # median chain must be past this fraction first
CAP_SUMMARY_THRESHOLD <- 0.05  # %-at-cap gate for the end-of-run advice line

#' cli routing depends only on the environment: an interactive session that
#' isn't rendering a knitr document. cli itself is a hard dependency, so there
#' is no package-availability check to make.
#' @noRd
should_use_cli_progress <- function() {
  interactive() && !isTRUE(getOption("knitr.in.progress"))
}

#' @noRd
resolve_progress_mode <- function(progress, refresh) {
  if (isTRUE(refresh <= 0L) || identical(progress, "none")) return("none")
  switch(
    progress,
    "auto" = if (should_use_cli_progress()) "cli" else "text",
    "cli" = "cli",
    "text" = "text",
    "none" = "none"
  )
}

#' @noRd
as_progress_num <- function(x, default = 0) {
  # `x[[1]]` keeps the is.na() guard scalar: a length > 1 input would otherwise
  # make `if (is.na(x))` error ("condition has length > 1").
  if (is.null(x) || length(x) == 0L || is.na(x[[1]])) return(default)
  as.numeric(x[[1]])
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
format_draw_count_compact <- function(x) {
  x <- as.integer(x)
  if (x >= 1000L) sprintf("%dk", x %/% 1000L) else as.character(x)
}

#' @noRd
format_divergence_status <- function(total_divs) {
  if (total_divs == 0L) return("div: 0")
  sprintf("%s div: %d", cli::symbol$warning, as.integer(total_divs))
}

#' @noRd
infer_tree_depth <- function(n_steps) {
  if (!is.finite(n_steps) || n_steps <= 0) return(NA_integer_)
  as.integer(ceiling(log2(n_steps + 1)))
}

#' @noRd
style_progress_status <- function(status, color = FALSE) {
  if (!isTRUE(color)) return(status)
  color_match <- function(s, pattern, fn) {
    m <- regexpr(pattern, s)
    if (m[[1]] > 0L) regmatches(s, m) <- fn(regmatches(s, m))
    s
  }
  status <- color_match(status, "[^ ]+ div: [1-9][0-9]*", cli::col_red)
  color_match(status, "[▲^] [0-9.]+ grad/draw", cli::col_yellow)
}

#' Accent the grad/draw token once the average crosses GRAD_HINT_THRESHOLD (an
#' absolute leapfrog-steps-per-draw count, not a fraction of the treedepth cap —
#' %-at-cap advice lives in the end summary). The `▲` glyph falls back to `^`
#' when the console can't render UTF-8; style_progress_status()'s regex matches
#' both.
#' @noRd
format_gradient_status <- function(avg_lf) {
  if (!is.finite(avg_lf)) return("- grad/draw")
  label <- sprintf("%.1f grad/draw", avg_lf)
  if (avg_lf >= GRAD_HINT_THRESHOLD) {
    accent <- if (cli::is_utf8_output()) "▲" else "^"
    paste(accent, label)
  } else {
    label
  }
}

#' @noRd
format_chain_draw_range <- function(snapshot) {
  if (length(snapshot) == 0L) return("")
  finished <- vapply(snapshot, function(s) as_progress_num(s$finished_draws), numeric(1))
  totals <- vapply(snapshot, function(s) as_progress_num(s$total_draws), numeric(1))
  total <- max(totals)
  min_f <- min(finished)
  max_f <- max(finished)
  total_str <- format_draw_count_compact(total)
  if (min_f == max_f) {
    sprintf("%s/%s", format_draw_count(min_f), total_str)
  } else {
    sprintf("%s-%s/%s", format_draw_count(min_f), format_draw_count(max_f), total_str)
  }
}

#' Gap-from-leader sparkline: one glyph per chain (in chain order), where the
#' glyph height is how far that chain trails the front-runner — NOT its absolute
#' progress. Chains that are together read as a flat baseline (all ▁); a laggard
#' shows up as a tall bar. Returns "" for single-chain runs. Opt-in `{spark}`
#' token (never in the default format); the percent-range token is `{spread}`.
#' @noRd
format_chain_spark <- function(snapshot) {
  if (length(snapshot) <= 1L) return("")
  finished <- vapply(snapshot, function(s) as_progress_num(s$finished_draws), numeric(1))
  totals <- vapply(snapshot, function(s) as_progress_num(s$total_draws), numeric(1))
  total <- max(totals)
  if (!is.finite(total) || total <= 0) return("")
  chain_vals <- vapply(snapshot, function(s) as.integer(as_progress_num(s$chain, NA_real_)), integer(1))
  ord <- order(ifelse(is.na(chain_vals), seq_along(snapshot), chain_vals))
  gaps <- max(finished) - finished[ord]
  ratio <- gaps / total
  glyphs <- strsplit("▁▂▃▄▅▆▇█", "")[[1]]
  lo <- 0.02  # deadzone: within 2% of the leader reads as caught up (flat)
  hi <- 0.20  # saturate the bar once a chain trails by 20% of total draws
  frac <- pmin(1, pmax(0, (ratio - lo) / (hi - lo)))
  level <- 1L + as.integer(round(frac * 7))
  paste(glyphs[level], collapse = "")
}

#' Per-chain progress fraction (finished/total) over *started* chains only.
#' Unstarted chains — queued when cores < num_chains — are excluded so they
#' don't read as a false spread at 0%.
#' @noRd
started_chain_fractions <- function(snapshot) {
  fracs <- vapply(snapshot, function(s) {
    if (!isTRUE(s$started)) return(NA_real_)
    total <- as_progress_num(s$total_draws)
    if (!is.finite(total) || total <= 0) return(NA_real_)
    as_progress_num(s$finished_draws) / total
  }, numeric(1))
  fracs[!is.na(fracs)]
}

#' Has the chain spread crossed the trigger? Over started chains:
#' `max - min >= SPREAD_TRIGGER_POINTS` (percentage points) and the median chain
#' is past `SPREAD_MEDIAN_FLOOR`. The median floor avoids firing in the noisy
#' opening laps before chains have settled.
#' @noRd
spread_triggered <- function(snapshot) {
  fracs <- started_chain_fractions(snapshot)
  if (length(fracs) < 2L) return(FALSE)
  (max(fracs) - min(fracs)) >= (SPREAD_TRIGGER_POINTS / 100) &&
    stats::median(fracs) >= SPREAD_MEDIAN_FLOOR
}

#' Percent-range spread token, e.g. "spread 23-78%" (ASCII hyphen, text-safe;
#' percent not draw counts to avoid colliding with the bar's current/total).
#' Empty until `active`; the callback latches `active` on once triggered, so it
#' renders for the rest of the run and collapses honestly (e.g. "spread 98-100%")
#' as chains converge.
#' @noRd
format_chain_spread <- function(snapshot, active = FALSE) {
  if (!isTRUE(active)) return("")
  fracs <- started_chain_fractions(snapshot)
  if (length(fracs) < 2L) return("")
  sprintf("spread %d-%d%%",
          as.integer(round(100 * min(fracs))),
          as.integer(round(100 * max(fracs))))
}

#' @noRd
format_chain_lag <- function(snapshot) {
  if (length(snapshot) <= 1L) return("")
  finished <- vapply(snapshot, function(s) as_progress_num(s$finished_draws), numeric(1))
  med <- median(finished)
  if (med == 0) return("")
  lag_idx <- which(finished < med * 0.9)
  if (length(lag_idx) == 0L) return("")
  chain_vals <- vapply(snapshot, function(s) as.integer(as_progress_num(s$chain, NA_real_)), integer(1))
  chain_labels <- ifelse(is.na(chain_vals), seq_along(snapshot), chain_vals)
  paste(paste0("c", chain_labels[lag_idx], collapse = ","), "slow")
}

#' @noRd
format_min_step <- function(snapshot) {
  steps <- vapply(snapshot, function(s) as_progress_num(s$step_size, NA_real_), numeric(1))
  finite_steps <- steps[is.finite(steps) & steps > 0]
  if (length(finite_steps) == 0L) return("")
  sprintf("%.3g", min(finite_steps))
}

#' @noRd
format_status_tokens <- function(snapshot, summary, max_treedepth, format_str,
                                 spread_active = FALSE) {
  result <- format_str
  if (grepl("{div}", format_str, fixed = TRUE))
    result <- gsub("{div}", format_divergence_status(summary$total_divergences),
                   result, fixed = TRUE)
  if (grepl("{grad}", format_str, fixed = TRUE))
    result <- gsub("{grad}", format_gradient_status(summary$avg_num_steps),
                   result, fixed = TRUE)
  if (grepl("{draws}", format_str, fixed = TRUE))
    result <- gsub("{draws}", format_chain_draw_range(snapshot), result, fixed = TRUE)
  if (grepl("{spread}", format_str, fixed = TRUE))
    result <- gsub("{spread}", format_chain_spread(snapshot, active = spread_active),
                   result, fixed = TRUE)
  if (grepl("{spark}", format_str, fixed = TRUE))
    result <- gsub("{spark}", format_chain_spark(snapshot), result, fixed = TRUE)
  if (grepl("{lag}", format_str, fixed = TRUE))
    result <- gsub("{lag}", format_chain_lag(snapshot), result, fixed = TRUE)
  if (grepl("{step}", format_str, fixed = TRUE))
    result <- gsub("{step}", format_min_step(snapshot), result, fixed = TRUE)
  # Collapse empty segments left by tokens that returned "". Multi-pass: a run
  # of three or more empty tokens (e.g. "{div} | {spread} | {spark}") needs more
  # than one sweep because each gsub consumes a pipe non-overlapping.
  while (grepl("\\|\\s*\\|", result)) {
    result <- gsub("\\s*\\|\\s*\\|", " |", result)
  }
  result <- gsub("^\\s*\\|\\s*", "", result)
  result <- gsub("\\s*\\|\\s*$", "", result)
  trimws(result)
}

#' @noRd
summarize_progress_snapshot <- function(snapshot, max_treedepth = 10L,
                                        num_warmup = 0L) {
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

  # `divergent_draws` indices from nuts-rs are warmup-inclusive and 0-based
  # (the `finished_draws` counter at divergence time, which already excludes
  # warmup divergences via nuts-rs's `!tuning` gate). Convert to the 1-indexed,
  # sample-phase-relative draw number used everywhere else in nutpieR.
  first_div <- NA_character_
  div_chains <- which(divergences > 0L)
  if (length(div_chains) > 0L) {
    candidates <- lapply(div_chains, function(i) {
      draws <- snapshot[[i]]$divergent_draws
      if (is.null(draws) || length(draws) == 0L) return(NULL)
      list(chain = chains[i],
           draw = min(as.integer(draws)) - as.integer(num_warmup) + 1L)
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
    sprintf("sample %d/%d", length(tuning) - num_tuning, length(tuning))
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
    format_gradient_status(avg_steps),
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
format_progress_value <- function(x, digits = NULL) {
  if (!is.finite(x)) return("NA")
  if (is.null(digits)) return(format_draw_count(x))
  sprintf(paste0("%.", digits, "f"), x)
}

#' @noRd
progress_supports_color <- function() {
  cli::num_ansi_colors() > 1L
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

#' Fraction of sampled draws that hit the max tree depth cap. Prefers the exact
#' `depth >= max_treedepth` test when a depth column exists; otherwise falls back
#' to `n_steps >= 2^max_treedepth - 1`. Returns `NA` when neither is available.
#' @noRd
fraction_at_treedepth_cap <- function(diagnostics, max_treedepth) {
  if (is.null(max_treedepth) || !is.finite(max_treedepth)) return(NA_real_)
  if (!is.null(diagnostics$depth)) {
    depth <- as.numeric(diagnostics$depth)
    at_cap <- depth >= as.numeric(max_treedepth)
  } else if (!is.null(diagnostics$n_steps)) {
    n_steps <- as.numeric(diagnostics$n_steps)
    at_cap <- n_steps >= (2^as.numeric(max_treedepth) - 1)
  } else {
    return(NA_real_)
  }
  at_cap <- at_cap[!is.na(at_cap)]
  if (length(at_cap) == 0L) return(NA_real_)
  mean(at_cap)
}

#' @noRd
print_sampling_diagnostic_summary <- function(diagnostics, num_chains, elapsed,
                                              max_treedepth = NULL) {
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

  if (total_divs > 0L) {
    cli::cli_alert_info(
      "Try increasing `target_accept`, inspecting pairs, or reparameterizing."
    )
  }

  cap_frac <- fraction_at_treedepth_cap(diagnostics, max_treedepth)
  if (is.finite(cap_frac) && cap_frac >= CAP_SUMMARY_THRESHOLD) {
    cap_pct <- sprintf("%d%%", as.integer(round(100 * cap_frac)))
    cli::cli_alert_info(
      "{cap_pct} of draws hit the max_treedepth cap — consider increasing `max_treedepth`."
    )
  }
  invisible(NULL)
}

#' Shared one-shot hint state for the cli and text callbacks. Each hint fires at
#' most once per run; the trigger flags live here so both callbacks fire each
#' hint exactly once. Emission is identical in both modes (see
#' [emit_progress_hint()]).
#' @noRd
new_progress_hints <- function() {
  env <- new.env(parent = emptyenv())
  env$warned_div <- FALSE
  env$warned_grad <- FALSE
  env$warned_spread <- FALSE
  env$spread_active <- FALSE
  # Per-chain late-warmup baselines for the grad/draw average, keyed by chain id.
  env$grad_baseline <- list()
  env
}

#' Record each started chain's `(total_num_steps, finished_draws)` the first
#' time it passes LATE_WARMUP_FRACTION of warmup. Anchoring here discards the
#' high-leapfrog early-warmup transient so the reported average reflects the
#' tuned sampler.
#' @noRd
update_grad_baselines <- function(hints, snapshot, num_warmup) {
  threshold <- LATE_WARMUP_FRACTION * as.numeric(num_warmup)
  for (s in snapshot) {
    chain <- as.integer(as_progress_num(s$chain, NA_real_))
    if (is.na(chain)) next
    finished <- as_progress_num(s$finished_draws)
    if (finished < threshold) next
    key <- as.character(chain)
    if (!is.null(hints$grad_baseline[[key]])) next
    hints$grad_baseline[[key]] <- list(
      total = as_progress_num(s$total_num_steps),
      finished = finished
    )
  }
  invisible(NULL)
}

#' Pooled grad/draw average over baselined chains:
#' `sum(total - base_total) / sum(finished - base_finished)`. Exact and
#' unbiased — no cap detection, no length-biased polling. Returns `NA` until at
#' least one baselined chain has produced a post-baseline draw.
#' @noRd
pooled_grad_per_draw <- function(hints, snapshot) {
  steps_delta <- 0
  draws_delta <- 0
  for (s in snapshot) {
    chain <- as.integer(as_progress_num(s$chain, NA_real_))
    if (is.na(chain)) next
    base <- hints$grad_baseline[[as.character(chain)]]
    if (is.null(base)) next
    steps_delta <- steps_delta + (as_progress_num(s$total_num_steps) - base$total)
    draws_delta <- draws_delta + (as_progress_num(s$finished_draws) - base$finished)
  }
  if (draws_delta <= 0) return(NA_real_)
  steps_delta / draws_delta
}

#' One-shot grad/draw hint (informational). Fires when the late-warmup-baseline
#' pooled average reaches GRAD_HINT_THRESHOLD. Tree depth is `round(log2(avg+1))`.
#' max_treedepth advice deliberately lives only in the end summary, where the
#' exact %-at-cap is known.
#' @noRd
maybe_grad_hint <- function(hints, avg) {
  if (hints$warned_grad || !is.finite(avg) || avg < GRAD_HINT_THRESHOLD) {
    return(invisible(NULL))
  }
  hints$warned_grad <- TRUE
  depth <- as.integer(round(log2(avg + 1)))
  emit_progress_hint(hints, "info", sprintf(
    paste0("grad/draw: high (~%d) gradient evaluations per draw (tree depth ~ %d) — ",
           "often a sign of difficult posterior geometry; worth a sanity check if unexpected."),
    as.integer(round(avg)), depth
  ))
}

#' One-shot chain-spread hint (cli only — in text mode the per-chain lines are
#' themselves the spread display). slowest/fastest are the started-chain min/max
#' percentages at trigger time.
#' @noRd
maybe_spread_hint <- function(hints, snapshot) {
  if (hints$warned_spread) return(invisible(NULL))
  fracs <- started_chain_fractions(snapshot)
  if (length(fracs) < 2L) return(invisible(NULL))
  hints$warned_spread <- TRUE
  emit_progress_hint(hints, "info", sprintf(
    paste0("spread: chain progress is uneven (slowest %d%%, fastest %d%%) — often ",
           "one chain adapted a smaller step size or is in a harder region of ",
           "the posterior. Adding to status line."),
    as.integer(round(100 * min(fracs))), as.integer(round(100 * max(fracs)))
  ))
}

#' Emit a one-time hint to stderr. `level` picks the leading glyph: "warning"
#' uses `cli::symbol$warning` (⚠, ASCII "!"), "info" uses `cli::symbol$info`
#' (ℹ, ASCII "i") — the same glyphs the status-line tokens use, so the bar and
#' the hints share one symbol vocabulary. Both cli and text modes go through
#' `message()`, keeping the whole progress stream on stderr and silenceable with
#' `suppressMessages()`.
#' @noRd
emit_progress_hint <- function(hints, level, msg) {
  sym <- switch(level, warning = cli::symbol$warning, info = cli::symbol$info)
  try(message(sym, " ", msg), silent = TRUE)
  invisible(NULL)
}

#' One-shot post-warmup divergence hint. `total_post_warmup_divs` is already
#' post-warmup-only — nuts-rs records divergences only outside tuning — so any
#' positive count means a post-warmup divergence has been observed.
#' @noRd
maybe_div_hint <- function(hints, total_post_warmup_divs) {
  if (hints$warned_div || total_post_warmup_divs <= 0L) return(invisible(NULL))
  hints$warned_div <- TRUE
  emit_progress_hint(
    hints, "warning",
    "div: divergent transitions detected — try increasing `target_accept` or reparameterizing."
  )
}

#' @noRd
make_cli_progress_callback <- function(num_chains, num_warmup, num_draws,
                                       max_treedepth = 10L,
                                       chain_format = NULL,
                                       id = NULL,
                                       update = NULL,
                                       done = NULL) {
  if (is.null(chain_format)) chain_format <- "{div} | {grad} | {spread}"
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
  hints <- new_progress_hints()
  callback_failed <- FALSE

  callback <- function(snapshot) {
    if (callback_failed) return(invisible(NULL))
    summary <- summarize_progress_snapshot(snapshot, max_treedepth = max_treedepth,
                                           num_warmup = num_warmup)
    last_summary <<- summary
    last_snapshot <<- snapshot
    total_now <- min(summary$total_finished, total_steps)

    maybe_div_hint(hints, summary$total_divergences)

    # Once chains pass late warmup, switch the {grad} token and the hint to the
    # baseline-adjusted average so the bar and the hint never disagree.
    update_grad_baselines(hints, snapshot, num_warmup)
    baseline_avg <- pooled_grad_per_draw(hints, snapshot)
    if (is.finite(baseline_avg)) summary$avg_num_steps <- baseline_avg
    maybe_grad_hint(hints, summary$avg_num_steps)

    # Spread: latch on once the started chains diverge enough, then keep the
    # {spread} token for the rest of the run (it collapses honestly as chains
    # converge). The hint fires once, at the latch.
    if (!hints$spread_active && spread_triggered(snapshot)) {
      hints$spread_active <- TRUE
      maybe_spread_hint(hints, snapshot)
    }

    raw_status <- format_status_tokens(snapshot, summary, max_treedepth, chain_format,
                                       spread_active = hints$spread_active)

    status <- style_progress_status(raw_status, color = progress_supports_color())
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
make_text_progress_callback <- function(num_chains, num_warmup, num_draws,
                                        max_treedepth = 10L,
                                        refresh = 10L,
                                        chain_format = NULL) {
  if (is.null(chain_format)) {
    chain_format <- "[{elapsed}] c{chain} {phase} {pct}  {draws}/{total} | {div} | {grad}"
  }
  last_printed <- rep(0L, num_chains)
  start_time <- proc.time()[["elapsed"]]
  hints <- new_progress_hints()

  callback <- function(snapshot) {
    if (length(snapshot) == 0L) return(invisible(NULL))
    elapsed_secs <- proc.time()[["elapsed"]] - start_time
    elapsed_str <- format_progress_time(elapsed_secs)

    # Post-warmup divergence hint (once). Each chain's `divergences` is already
    # post-warmup-only, so the sum across chains is the post-warmup total.
    total_divs <- sum(vapply(
      snapshot, function(s) as.integer(as_progress_num(s$divergences)), integer(1)
    ))
    maybe_div_hint(hints, total_divs)

    # grad/draw hint (once), from the same late-warmup baseline the cli bar uses.
    update_grad_baselines(hints, snapshot, num_warmup)
    maybe_grad_hint(hints, pooled_grad_per_draw(hints, snapshot))

    for (s in snapshot) {
      chain_idx <- as.integer(as_progress_num(s$chain, NA_real_))
      if (is.na(chain_idx) || chain_idx < 1L || chain_idx > num_chains) next

      finished <- as.integer(as_progress_num(s$finished_draws))
      since_last <- finished - last_printed[chain_idx]
      if (since_last < as.integer(refresh)) next
      last_printed[chain_idx] <<- finished

      total <- as.integer(as_progress_num(s$total_draws))
      pct <- if (total > 0L) {
        sprintf("%d%%", as.integer(round(100 * finished / total)))
      } else {
        "0%"
      }
      phase_str <- if (isTRUE(s$tuning)) "warmup" else "sample"
      total_steps_chain <- as_progress_num(s$total_num_steps)
      avg_lf <- if (finished > 0L) total_steps_chain / finished else NA_real_
      div_count <- as.integer(as_progress_num(s$divergences))

      line <- chain_format
      line <- gsub("{chain}", chain_idx, line, fixed = TRUE)
      line <- gsub("{phase}", phase_str, line, fixed = TRUE)
      line <- gsub("{pct}", pct, line, fixed = TRUE)
      line <- gsub("{draws}", format_draw_count(finished), line, fixed = TRUE)
      line <- gsub("{total}", format_draw_count(total), line, fixed = TRUE)
      line <- gsub("{elapsed}", elapsed_str, line, fixed = TRUE)
      line <- gsub("{div}", format_divergence_status(div_count), line, fixed = TRUE)
      line <- gsub("{grad}", format_gradient_status(avg_lf), line, fixed = TRUE)

      # stderr via message() so text lines interleave correctly with hints and
      # the end summary, and suppressMessages() silences the whole stream.
      message(line)
    }
    invisible(NULL)
  }

  attr(callback, "finish") <- function() invisible(NULL)
  callback
}

#' @noRd
finish_progress_callback <- function(callback) {
  finish <- attr(callback, "finish")
  if (is.function(finish)) finish()
  invisible(NULL)
}
