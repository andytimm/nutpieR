#' Convert a flat draws matrix to posterior::draws_array
#'
#' @param flat_matrix Matrix with (n_draws * n_chains) rows and n_params
#'   columns. Rows are ordered by chain (chain 1 rows first, then chain 2, etc).
#' @param n_draws Number of draws per chain.
#' @param n_chains Number of chains.
#' @return A `posterior::draws_array` object.
#' @noRd
matrix_to_draws_array <- function(flat_matrix, n_draws, n_chains) {
  param_names <- dot_to_bracket(colnames(flat_matrix))
  arr <- array(flat_matrix,
    dim = c(n_draws, n_chains, ncol(flat_matrix)),
    dimnames = list(
      iteration = seq_len(n_draws),
      chain = seq_len(n_chains),
      variable = param_names
    )
  )
  posterior::as_draws_array(arr)
}

#' Convert BridgeStan dot-indexed names to bracket notation
#'
#' BridgeStan returns `beta.1.2` for a matrix parameter; Stan convention is
#' `beta[1,2]`. Scalar parameters (no trailing digits) are left unchanged.
#' @noRd
dot_to_bracket <- function(names) {
  vapply(names, function(nm) {
    # Match trailing .digit(.digit)* sequence
    m <- regexpr("\\.(\\d+(?:\\.\\d+)*)$", nm, perl = TRUE)
    if (m == -1L) return(nm)
    prefix <- substr(nm, 1, m - 1)
    indices <- gsub("\\.", ",", substr(nm, m + 1, nchar(nm)))
    paste0(prefix, "[", indices, "]")
  }, character(1), USE.NAMES = FALSE)
}
