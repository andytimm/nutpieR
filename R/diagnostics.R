#' Extract sampler diagnostics from nutpie draws
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()].
#' @return A `nutpie_diagnostics` object (a named list with a print method).
#'   Fields include `diverging`, `depth`, `energy`, `energy_error`, `logp`,
#'   `n_steps`, `step_size_bar`, and `mean_tree_accept`. When
#'   `store_divergences = TRUE` was used, additional list columns
#'   `divergence_start`, `divergence_end`, `divergence_momentum`, and
#'   `divergence_start_gradient` are included. When `store_mass_matrix = TRUE`
#'   was used, a `mass_matrix_inv` list column is included.
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
  n <- length(x$diverging)
  num_chains <- attr(x, "num_chains") %||% 1L
  n_per_chain <- n %/% num_chains

  n_div <- sum(x$diverging)
  max_depth <- max(x$depth)
  mean_accept <- mean(x$mean_tree_accept)
  step_sizes <- x$step_size_bar[seq(n_per_chain, n, by = n_per_chain)]

  cat("Sampler diagnostics\n")
  cat(sprintf("  Draws:         %d (%d per chain, %d chains)\n",
              n, n_per_chain, num_chains))
  cat(sprintf("  Divergences:   %d", n_div))
  if (n_div > 0) cat(sprintf(" (%.1f%%)", 100 * n_div / n))
  cat("\n")
  cat(sprintf("  Max treedepth: %d\n", max_depth))
  cat(sprintf("  Mean accept:   %.3f\n", mean_accept))
  cat(sprintf("  Step size:     %s\n",
              paste(sprintf("%.4f", step_sizes), collapse = ", ")))

  if (n_div > 0) {
    cat("\n")
    cat("  Warning: divergent transitions detected. Consider increasing\n")
    cat("  target_accept or reparameterizing the model.\n")
  }

  cat("\nUse `str(attr(draws, \"diagnostics\"))` for raw diagnostic vectors.\n")
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
