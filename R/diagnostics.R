#' Extract sampler diagnostics from nutpie draws
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()].
#' @return A named list of diagnostic vectors (one value per draw, ordered
#'   chain-contiguous). Fields include `diverging`, `depth`, `energy`,
#'   `energy_error`, `logp`, `n_steps`, `step_size_bar`, and
#'   `mean_tree_accept`. When `store_divergences = TRUE` was used, additional
#'   list columns `divergence_start`, `divergence_end`,
#'   `divergence_momentum`, and `divergence_start_gradient` are included.
#'   When `store_mass_matrix = TRUE` was used, a `mass_matrix_inv` list
#'   column is included.
#' @export
nutpie_diagnostics <- function(draws) {
  diag <- attr(draws, "diagnostics")
  if (is.null(diag)) {
    stop("No diagnostics found. Was this object created by nutpie_sample()?",
         call. = FALSE)
  }
  diag
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
