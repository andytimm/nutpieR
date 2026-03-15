#' Extract sampler diagnostics from nutpie draws
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()].
#' @return A named list of diagnostic vectors (one value per draw, ordered
#'   chain-contiguous). Fields include `diverging`, `depth`, `energy`,
#'   `energy_error`, `logp`, `n_steps`, `step_size_bar`, and
#'   `mean_tree_accept`.
#' @export
nutpie_diagnostics <- function(draws) {
  diag <- attr(draws, "diagnostics")
  if (is.null(diag)) {
    stop("No diagnostics found. Was this object created by nutpie_sample()?",
         call. = FALSE)
  }
  diag
}
