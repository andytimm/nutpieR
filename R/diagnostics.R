# Bring the upstream nuts-rs `chain` (0-indexed) and `draw` (cumulative
# 1-indexed) onto `posterior::draws_array` conventions: `chain` 1-indexed in
# 1:num_chains, `draw` 1-indexed within phase (1..num_draws or 1..num_warmup).
reindex_diagnostics <- function(diag, num_warmup, phase = c("sample", "warmup")) {
  if (is.null(diag)) return(diag)
  phase <- match.arg(phase)
  if (!is.null(diag$chain)) {
    diag$chain <- diag$chain + 1L
  }
  if (!is.null(diag$draw) && identical(phase, "sample")) {
    diag$draw <- diag$draw - as.integer(num_warmup)
  }
  diag
}

# Upstream nuts-rs uses "tune"; we align with R / Stan / cmdstanr "warmup"
# on the way out. tryCatch falls back to the original string so a future
# schema bump that breaks fromJSON doesn't lose the attribute.
rename_sampler_config <- function(json_str) {
  tryCatch({
    cfg <- jsonlite::fromJSON(json_str, simplifyVector = FALSE)
    if (!is.null(cfg$num_tune)) {
      cfg$num_warmup <- cfg$num_tune
      cfg$num_tune <- NULL
    }
    jsonlite::toJSON(cfg, auto_unbox = TRUE)
  }, error = function(e) json_str)
}

# Failure-flag thresholds. One-directional: a trip proves a problem; staying
# under proves nothing (so we flag, never certify). R-hat from Vehtari et al.
# 2021 / posterior; ESS floor from the same; E-BFMI default matches CmdStanR's
# `check_ebfmi`. EBFMI_THRESHOLD lives in progress.R (shared via the namespace).
RHAT_THRESHOLD <- 1.01
ESS_THRESHOLD  <- 400

#' Per-chain E-BFMI using the grounded CmdStanR formula (NOT a fraction-of-draws
#' heuristic). For each chain, with energy vector `e`:
#' `(sum(diff(e)^2) / length(e)) / var(e)`. Returns a numeric vector named by
#' chain id, `NA` per chain when energy is absent / has NAs / has < 3 draws /
#' has zero variance. Returns `NULL` when there is no energy or chain field at
#' all. Mirrors `cmdstanr::ebfmi` guards.
#' @noRd
ebfmi_per_chain <- function(diagnostics) {
  if (is.null(diagnostics) || is.null(diagnostics$energy) ||
      is.null(diagnostics$chain)) {
    return(NULL)
  }
  energy <- as.numeric(diagnostics$energy)
  chain <- as.integer(diagnostics$chain)
  chains <- sort(unique(chain))
  out <- vapply(chains, function(ch) {
    e <- energy[chain == ch]
    if (length(e) < 3L || anyNA(e)) return(NA_real_)
    v <- stats::var(e)
    if (!is.finite(v) || v <= 0) return(NA_real_)
    (sum(diff(e)^2) / length(e)) / v
  }, numeric(1))
  stats::setNames(out, as.character(chains))
}

#' Cross-chain convergence extrema for the diagnostics print. Runs
#' `posterior::summarise_draws()` with R-hat / bulk-ESS / tail-ESS and returns
#' the worst value of each plus the offending variable name. R-hat needs >= 2
#' chains, so `max_rhat`/`max_rhat_var` are `NULL` for single-chain runs.
#' Computed lazily (print-only) so attaching the draws array as an attribute
#' stays cheap. Returns `NULL` if `draws` is `NULL`.
#' @noRd
worst_rhat_ess <- function(draws) {
  if (is.null(draws)) return(NULL)
  num_chains <- tryCatch(posterior::nchains(draws),
                         error = function(e) NA_integer_)
  measures <- list(ess_bulk = posterior::ess_bulk,
                   ess_tail = posterior::ess_tail)
  if (isTRUE(num_chains >= 2L)) {
    measures <- c(list(rhat = posterior::rhat), measures)
  }
  s <- do.call(posterior::summarise_draws, c(list(draws), measures))
  pick <- function(col, fn) {
    vals <- s[[col]]
    if (is.null(vals) || all(is.na(vals))) return(list(val = NULL, var = NULL))
    i <- fn(vals)
    list(val = vals[[i]], var = as.character(s$variable[[i]]))
  }
  rhat <- if ("rhat" %in% names(s)) pick("rhat", which.max) else list(val = NULL, var = NULL)
  bulk <- pick("ess_bulk", which.min)
  tail <- pick("ess_tail", which.min)
  list(
    max_rhat = rhat$val, max_rhat_var = rhat$var,
    min_ess_bulk = bulk$val, min_ess_bulk_var = bulk$var,
    min_ess_tail = tail$val, min_ess_tail_var = tail$var
  )
}

#' Shared divergence-severity wording so the post-run summary and the
#' diagnostics print speak with one voice. `severe` (any per-chain or pooled
#' divergence share >= `DIV_SEVERE_THRESHOLD`) escalates from a tuning hint to
#' a geometry warning. Returns a plain string (no glue braces) — each surface
#' emits it with its own mechanism.
#' @noRd
format_div_severity_msg <- function(total_divs, div_frac, severe) {
  plural <- if (identical(as.integer(total_divs), 1L)) "" else "s"
  div_pct <- if (is.finite(div_frac)) sprintf("%.1f%%", 100 * div_frac) else "NA"
  head <- sprintf("%d divergent transition%s (%s of post-warmup draws)",
                  as.integer(total_divs), plural, div_pct)
  if (isTRUE(severe)) {
    paste0(head, "; results are not reliable. This usually means the model ",
           "geometry is exposing a deeper problem, not just a tuning issue. ",
           "Check parameterization, constraints, and priors before relying on ",
           "the fit.")
  } else {
    paste0(head, ". Try increasing `target_accept`, inspecting pairs plots, ",
           "or reparameterizing.")
  }
}

#' Shared "is this fit's divergence count severe?" test. `div_frac` is the
#' pooled post-warmup divergence share; `per_chain_frac` a per-chain vector
#' (NA-tolerant). Severe when either crosses `DIV_SEVERE_THRESHOLD` — the same
#' rule the post-run summary and the diagnostics print both apply, so they
#' can't disagree on whether to escalate from a tuning hint to a geometry
#' warning.
#' @noRd
divergence_is_severe <- function(div_frac, per_chain_frac) {
  is.finite(div_frac) &&
    (div_frac >= DIV_SEVERE_THRESHOLD ||
       any(per_chain_frac >= DIV_SEVERE_THRESHOLD, na.rm = TRUE))
}

#' Shared E-BFMI warning sentence. Returns the "N of M chains ..." string when
#' any chain falls below `EBFMI_THRESHOLD`, else `NULL`, so the post-run summary
#' and the diagnostics print never drift on the number (and threshold) users
#' quote. The returned string carries no glue braces; each surface emits it.
#' @noRd
ebfmi_warning_msg <- function(ebfmi) {
  if (is.null(ebfmi)) return(NULL)
  n_low <- sum(is.finite(ebfmi) & ebfmi < EBFMI_THRESHOLD)
  if (n_low == 0L) return(NULL)
  sprintf(
    paste0("%d of %d chains had an E-BFMI below 0.3 \u2014 the posterior may have ",
           "heavy tails the sampler explores inefficiently. Consider ",
           "reparameterizing."),
    n_low, length(ebfmi)
  )
}

#' Extract sampler diagnostics from nutpie draws
#'
#' Diagnostics are extracted directly from the nuts-rs sample-stats schema, so
#' the exact set of fields depends on the installed nuts-rs version and the
#' sampling options used. Count fields (`depth`, `n_steps`, `chain`, `draw`,
#' `index_in_trajectory`) are returned as R `integer` when every value fits
#' in `i32`, else as `numeric`; floating-point fields (`logp`, `energy`,
#' `step_size`, etc.) are always `numeric`. `NA`s use the matching sentinel
#' (`NA_integer_` / `NA_real_`).
#'
#' @section Indexing conventions:
#' `chain` is 1-indexed in `1:num_chains`. `draw` is 1-indexed in
#' `1:num_draws` for post-warmup diagnostics and `1:num_warmup` for warmup
#' diagnostics (returned via [nutpie_warmup_diagnostics()]). Matches
#' `posterior::draws_array` conventions, so a `data.frame` of diagnostics
#' joins cleanly against draws indexed by `(chain, iteration)`.
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()].
#' @return A `nutpie_diagnostics` object (a named list with a print method).
#'   Commonly available scalar fields: `diverging`, `tuning`, `maxdepth_reached`
#'   (logical); `depth`, `n_steps`, `chain`, `draw`, `index_in_trajectory`
#'   (integer when fits in `i32`, else numeric); `logp`, `energy`,
#'   `energy_error`, `step_size`, `step_size_bar`, `mean_tree_accept`,
#'   `mean_tree_accept_sym` (numeric).
#'   Wide fields (one row per draw): when `store_unconstrained = TRUE`,
#'   `unconstrained_draw` (`NA` rows where unrecorded); when
#'   `store_gradient = TRUE`, `gradient` (`NA` rows where unrecorded); when
#'   `store_mass_matrix = TRUE` **and** `save_warmup = TRUE`,
#'   `mass_matrix_inv` (and `mass_matrix_eigvals` / `mass_matrix_stds` when
#'   reported), with the most recently recorded value carried forward into
#'   draws between updates — the inverse mass matrix is piecewise-constant
#'   between adapter steps, not undefined. Requires `save_warmup = TRUE`
#'   because adaptation (the only time the matrix changes) occurs during
#'   warmup; without warmup draws in the trace there is no value to carry
#'   forward.
#'   These surface as `(n_draws * n_chains, ndim_unc)` numeric matrices when
#'   every recorded row has the same width; mixed-width columns fall back
#'   to a list of numeric vectors (one per draw, `NULL` when not recorded).
#'   With `store_divergences = TRUE`: `divergence_start`, `divergence_end`,
#'   `divergence_momentum`, `divergence_start_gradient` (lists, only
#'   present when at least one draw diverged).
#' @examples
#' \dontrun{
#' draws <- nutpie_sample(model, data = dat, num_draws = 1000, num_chains = 4)
#' diag <- nutpie_diagnostics(draws)
#' diag                               # printed summary
#' sum(diag$diverging)                # divergence count
#' max(diag$depth)                    # peak treedepth across all draws
#' }
#' @export
nutpie_diagnostics <- function(draws) {
  diag <- attr(draws, "diagnostics")
  if (is.null(diag)) {
    stop("No diagnostics found. Was this object created by nutpie_sample()?",
         call. = FALSE)
  }
  num_chains <- attr(draws, "num_chains") %||% dim(draws)[[2]] %||% 1L
  out <- structure(diag, class = "nutpie_diagnostics", num_chains = num_chains)
  # Reference (not a copy) so print() can compute R-hat/ESS lazily; nothing is
  # computed here, so field access stays cheap.
  attr(out, "draws") <- draws
  out
}

#' @export
print.nutpie_diagnostics <- function(x, ...) {
  n <- length(x$diverging %||% x[[1]])
  num_chains <- attr(x, "num_chains") %||% 1L
  n_per_chain <- n %/% num_chains

  cat("Sampler diagnostics\n")
  cat(sprintf("  Draws:         %d (%d per chain, %d chains)\n",
              n, n_per_chain, num_chains))

  if (!is.null(x$diverging)) {
    n_div <- sum(x$diverging, na.rm = TRUE)
    cat(sprintf("  Divergences:   %d", n_div))
    if (n_div > 0) cat(sprintf(" (%.1f%%)", 100 * n_div / n))
    cat("\n")
  }
  if (!is.null(x$maxdepth_reached)) {
    n_md <- sum(x$maxdepth_reached, na.rm = TRUE)
    cat(sprintf("  Max-treedepth hits: %d", n_md))
    if (n_md > 0) cat(sprintf(" (%.1f%%)", 100 * n_md / n))
    cat("\n")
  } else if (!is.null(x$depth)) {
    cat(sprintf("  Max treedepth: %d\n", max(x$depth, na.rm = TRUE)))
  }
  if (!is.null(x$mean_tree_accept)) {
    cat(sprintf("  Mean accept:   %.3f\n", mean(x$mean_tree_accept, na.rm = TRUE)))
  }
  if (!is.null(x$step_size_bar)) {
    step_sizes <- x$step_size_bar[seq(n_per_chain, n, by = n_per_chain)]
    cat(sprintf("  Step size:     %s\n",
                paste(sprintf("%.4f", step_sizes), collapse = ", ")))
  }

  # ── Cross-chain failure flags (the post-completion diagnostics) ───────────
  # Numbers shown every time (like Mean accept / Step size); a warning line
  # fires only on a threshold trip, naming the offending parameter. All three
  # checks degrade quietly when their preconditions are unmet.
  rhat_ess <- worst_rhat_ess(attr(x, "draws"))
  if (!is.null(rhat_ess)) {
    if (!is.null(rhat_ess$max_rhat)) {
      cat(sprintf("  Max R-hat:     %.2f (%s)\n",
                  rhat_ess$max_rhat, rhat_ess$max_rhat_var))
    } else {
      cat("  Max R-hat:     - (needs >= 2 chains)\n")
    }
    if (!is.null(rhat_ess$min_ess_bulk)) {
      cat(sprintf("  Min Bulk-ESS:  %.0f (%s)\n",
                  rhat_ess$min_ess_bulk, rhat_ess$min_ess_bulk_var))
    }
    if (!is.null(rhat_ess$min_ess_tail)) {
      cat(sprintf("  Min Tail-ESS:  %.0f (%s)\n",
                  rhat_ess$min_ess_tail, rhat_ess$min_ess_tail_var))
    }
  }

  ebfmi <- ebfmi_per_chain(x)
  finite_ebfmi <- if (!is.null(ebfmi)) ebfmi[is.finite(ebfmi)] else numeric()
  if (length(finite_ebfmi) > 0L) {
    cat(sprintf("  Min E-BFMI:    %.2f\n", min(finite_ebfmi)))
  }

  # Divergence flag: same severity-gated vocabulary as the post-run summary.
  if (!is.null(x$diverging)) {
    n_div <- sum(x$diverging, na.rm = TRUE)
    if (n_div > 0L) {
      div_frac <- n_div / n
      per_chain_frac <- NA_real_
      if (!is.null(x$chain)) {
        ch <- as.integer(x$chain)
        dv <- as.logical(x$diverging)
        per_chain_frac <- vapply(sort(unique(ch)), function(c) {
          mean(dv[ch == c], na.rm = TRUE)
        }, numeric(1))
      }
      severe <- divergence_is_severe(div_frac, per_chain_frac)
      cli::cli_alert_warning("{format_div_severity_msg(n_div, div_frac, severe)}")
    }
  }

  # Convergence flag (R-hat / ESS): combine into one line, naming offenders.
  if (!is.null(rhat_ess)) {
    rhat_trip <- !is.null(rhat_ess$max_rhat) && is.finite(rhat_ess$max_rhat) &&
      rhat_ess$max_rhat > RHAT_THRESHOLD
    bulk_trip <- !is.null(rhat_ess$min_ess_bulk) &&
      is.finite(rhat_ess$min_ess_bulk) && rhat_ess$min_ess_bulk < ESS_THRESHOLD
    tail_trip <- !is.null(rhat_ess$min_ess_tail) &&
      is.finite(rhat_ess$min_ess_tail) && rhat_ess$min_ess_tail < ESS_THRESHOLD
    if (rhat_trip || bulk_trip || tail_trip) {
      parts <- character()
      vars <- character()
      if (rhat_trip) {
        parts <- c(parts, sprintf("R-hat %.2f > %s",
                                  rhat_ess$max_rhat, format(RHAT_THRESHOLD)))
        vars <- c(vars, rhat_ess$max_rhat_var)
      }
      # When bulk and tail trip on the same parameter, fold them into one
      # clause ("Bulk/Tail-ESS 6/168 < 400") rather than two "and"s.
      if (bulk_trip && tail_trip &&
          identical(rhat_ess$min_ess_bulk_var, rhat_ess$min_ess_tail_var)) {
        parts <- c(parts, sprintf("Bulk/Tail-ESS %.0f/%.0f < %d",
                                  rhat_ess$min_ess_bulk, rhat_ess$min_ess_tail,
                                  ESS_THRESHOLD))
        vars <- c(vars, rhat_ess$min_ess_bulk_var)
      } else {
        if (bulk_trip) {
          parts <- c(parts, sprintf("Bulk-ESS %.0f < %d",
                                    rhat_ess$min_ess_bulk, ESS_THRESHOLD))
          vars <- c(vars, rhat_ess$min_ess_bulk_var)
        }
        if (tail_trip) {
          parts <- c(parts, sprintf("Tail-ESS %.0f < %d",
                                    rhat_ess$min_ess_tail, ESS_THRESHOLD))
          vars <- c(vars, rhat_ess$min_ess_tail_var)
        }
      }
      uvars <- unique(vars)
      var_label <- paste(sprintf("`%s`", uvars), collapse = ", ")
      # No summarize_draws pointer here — the always-shown footer carries it
      # once, so a flagged fit doesn't repeat it.
      msg <- sprintf("%s for %s: chains may not have mixed.",
                     paste(parts, collapse = " and "), var_label)
      cli::cli_alert_warning("{msg}")
    }
  }

  # E-BFMI flag: independent of geometry; energy problems coexist with clean
  # divergence/treedepth.
  ebfmi_msg <- ebfmi_warning_msg(ebfmi)
  if (!is.null(ebfmi_msg)) {
    cli::cli_alert_warning("{ebfmi_msg}")
  }

  cat("\nFor the full per-parameter table, see posterior::summarize_draws(draws).\n")
  invisible(x)
}

#' Extract warmup draws from nutpie output
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()] with
#'   `save_warmup = TRUE`.
#' @return A `posterior::draws_array` containing the warmup draws, or `NULL`
#'   if warmup draws were not saved.
#' @examples
#' \dontrun{
#' draws <- nutpie_sample(model, data = dat, save_warmup = TRUE)
#' warmup <- nutpie_warmup_draws(draws)
#' posterior::summarize_draws(warmup)
#' }
#' @export
nutpie_warmup_draws <- function(draws) {
  warmup <- attr(draws, "warmup_draws")
  if (is.null(warmup)) {
    stop("No warmup draws found. Did you use save_warmup = TRUE?",
         call. = FALSE)
  }
  warmup
}

#' Extract warmup diagnostics from nutpie output
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()] with
#'   `save_warmup = TRUE`.
#' @return A named list of diagnostic vectors for the warmup phase. `chain`
#'   is 1-indexed and `draw` is 1-indexed in `1:num_warmup`; see
#'   [nutpie_diagnostics()] for the full indexing convention.
#' @examples
#' \dontrun{
#' draws <- nutpie_sample(model, data = dat, save_warmup = TRUE)
#' wd <- nutpie_warmup_diagnostics(draws)
#' max(wd$depth)                       # warmup treedepth peak
#' }
#' @export
nutpie_warmup_diagnostics <- function(draws) {
  diag <- attr(draws, "warmup_diagnostics")
  if (is.null(diag)) {
    stop("No warmup diagnostics found. Did you use save_warmup = TRUE?",
         call. = FALSE)
  }
  diag
}

#' NUTS sampler parameters in bayesplot's long format
#'
#' Reshapes the diagnostics from [nutpie_diagnostics()] into the four-column
#' long-format `data.frame` that bayesplot's NUTS plotting helpers (e.g.
#' `bayesplot::mcmc_pairs(np = ...)`, `bayesplot::mcmc_nuts_energy()`)
#' expect. Names match Stan's CSV convention (`accept_stat__`,
#' `divergent__`, `treedepth__`, `n_leapfrog__`, `stepsize__`,
#' `energy__`). Other bayesplot NUTS helpers (e.g.
#' `mcmc_nuts_divergence()`, `mcmc_nuts_acceptance()`) additionally
#' need a per-draw `lp` data frame, which this helper does not produce.
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()].
#' @return A `data.frame` with columns:
#'   \describe{
#'     \item{`Chain`}{Integer chain index (1-indexed).}
#'     \item{`Iteration`}{Integer post-warmup draw index (1-indexed
#'       within chain, in `1:num_draws`).}
#'     \item{`Parameter`}{Character; one of `"accept_stat__"`,
#'       `"divergent__"`, `"treedepth__"`, `"n_leapfrog__"`,
#'       `"stepsize__"`, `"energy__"`.}
#'     \item{`Value`}{Numeric value of the corresponding diagnostic.}
#'   }
#'   The data frame has `num_draws * num_chains * 6` rows.
#' @seealso [bayesplot::mcmc_pairs()] for the most common consumer of this
#'   format. [nutpie_diagnostics()] for the raw diagnostics.
#' @examples
#' \dontrun{
#' draws <- nutpie_sample(model, data = dat, num_chains = 4)
#' np <- nutpie_nuts_params(draws)
#' bayesplot::mcmc_pairs(draws, np = np, pars = c("mu", "tau"))
#' }
#' @export
nutpie_nuts_params <- function(draws) {
  diag <- nutpie_diagnostics(draws)
  chain <- as.integer(diag$chain)
  iter  <- as.integer(diag$draw)
  n <- length(chain)
  params <- c("accept_stat__", "divergent__", "treedepth__",
              "n_leapfrog__", "stepsize__", "energy__")
  # A backend may not populate every field. Fill absent fields with NA of the
  # right length so `Value` stays aligned with `Parameter` instead of recycling
  # a short vector into mislabeled columns.
  or_na <- function(x) if (is.null(x)) rep(NA_real_, n) else as.numeric(x)
  values <- c(
    or_na(diag$mean_tree_accept),
    or_na(diag$diverging),
    or_na(diag$depth),
    or_na(diag$n_steps),
    or_na(diag$step_size),
    or_na(diag$energy)
  )

  data.frame(
    Chain     = rep(chain,  times = length(params)),
    Iteration = rep(iter,   times = length(params)),
    Parameter = rep(params, each  = n),
    Value     = values,
    stringsAsFactors = FALSE
  )
}
