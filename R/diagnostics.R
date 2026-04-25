#' Extract sampler diagnostics from nutpie draws
#'
#' Diagnostics are extracted directly from the nuts-rs sample-stats schema, so
#' the exact set of fields depends on the installed nuts-rs version and the
#' sampling options used. All numeric fields are returned as R `double`, even
#' when the underlying Arrow column is integer-typed (depth, n_steps, chain,
#' draw, etc.) — this avoids silent truncation.
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()].
#' @return A `nutpie_diagnostics` object (a named list with a print method).
#'   Commonly available scalar fields: `diverging`, `tuning`, `maxdepth_reached`
#'   (logical); `depth`, `n_steps`, `chain`, `draw`, `index_in_trajectory`,
#'   `logp`, `energy`, `energy_error`, `step_size`, `step_size_bar`,
#'   `mean_tree_accept`, `mean_tree_accept_sym` (numeric).
#'   List-valued fields (one entry per draw, `NULL` when not recorded):
#'   `unconstrained_draw`, `gradient`. With `store_mass_matrix = TRUE`:
#'   `mass_matrix_inv`. With `store_divergences = TRUE`: `divergence_start`,
#'   `divergence_end`, `divergence_momentum`, `divergence_start_gradient`
#'   (only present when at least one draw diverged).
#' @export
nutpie_diagnostics <- function(draws) {
  diag <- attr(draws, "diagnostics")
  if (is.null(diag)) {
    stop("No diagnostics found. Was this object created by nutpie_sample()?",
         call. = FALSE)
  }
  num_chains <- attr(draws, "num_chains") %||% dim(draws)[[2]] %||% 1L
  structure(diag, class = "nutpie_diagnostics", num_chains = num_chains)
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
    n_div <- sum(x$diverging)
    cat(sprintf("  Divergences:   %d", n_div))
    if (n_div > 0) cat(sprintf(" (%.1f%%)", 100 * n_div / n))
    cat("\n")
  }
  if (!is.null(x$maxdepth_reached)) {
    n_md <- sum(x$maxdepth_reached)
    cat(sprintf("  Max-treedepth hits: %d", n_md))
    if (n_md > 0) cat(sprintf(" (%.1f%%)", 100 * n_md / n))
    cat("\n")
  } else if (!is.null(x$depth)) {
    cat(sprintf("  Max treedepth: %d\n", max(x$depth)))
  }
  if (!is.null(x$mean_tree_accept)) {
    cat(sprintf("  Mean accept:   %.3f\n", mean(x$mean_tree_accept)))
  }
  if (!is.null(x$step_size_bar)) {
    step_sizes <- x$step_size_bar[seq(n_per_chain, n, by = n_per_chain)]
    cat(sprintf("  Step size:     %s\n",
                paste(sprintf("%.4f", step_sizes), collapse = ", ")))
  }

  if (!is.null(x$diverging) && sum(x$diverging) > 0) {
    cat("\n")
    cat("  Warning: divergent transitions detected. Consider increasing\n")
    cat("  target_accept or reparameterizing the model.\n")
  }

  cat(sprintf("\nAvailable fields: %s\n",
              paste(names(x), collapse = ", ")))
  cat("Use `str(nutpie_diagnostics(draws))` for raw diagnostic vectors.\n")
  invisible(x)
}

#' Extract warmup draws from nutpie output
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()] with
#'   `save_warmup = TRUE`.
#' @return A `posterior::draws_array` containing the warmup draws, or `NULL`
#'   if warmup draws were not saved.
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
#' @return A named list of diagnostic vectors for the warmup phase.
#' @export
nutpie_warmup_diagnostics <- function(draws) {
  diag <- attr(draws, "warmup_diagnostics")
  if (is.null(diag)) {
    stop("No warmup diagnostics found. Did you use save_warmup = TRUE?",
         call. = FALSE)
  }
  diag
}

`%||%` <- function(a, b) if (is.null(a)) b else a
