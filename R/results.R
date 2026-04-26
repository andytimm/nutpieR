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

#' Block-level prefixes from a vector of bridgestan dot-indexed names
#'
#' `beta.1.2` and `beta.2.1` both collapse to `beta`; scalar names pass
#' through unchanged.
#' @noRd
block_prefixes <- function(dot_names) unique(sub("\\..*", "", dot_names))

#' Resolve `pars` / `include` to bridgestan `(include_tp, include_gq)` flags
#'
#' Skipping the TP / GQ slice in `param_constrain` saves both the per-draw
#' allocation and the Stan-side compute (the GQ block's `*_rng` calls).
#' GQ may reference TP, so we conservatively force TP on whenever GQ is on.
#'
#' This is also the single place where unknown `pars` names are validated;
#' downstream `resolve_keep_indices` trusts the input.
#' @noRd
resolve_constrain_flags <- function(handle, pars, include) {
  if (is.null(pars)) return(list(include_tp = TRUE, include_gq = TRUE))

  block <- block_prefixes(bs_block_names(handle))
  block_tp <- block_prefixes(bs_block_tp_names(handle))
  full <- block_prefixes(bs_full_names(handle))

  bad <- setdiff(pars, full)
  if (length(bad) > 0) {
    stop("Unknown parameter(s): ", paste(bad, collapse = ", "),
         "\nAvailable: ", paste(full, collapse = ", "),
         call. = FALSE)
  }

  tp_only <- setdiff(block_tp, block)
  gq_only <- setdiff(full, block_tp)

  kept <- if (include) pars else setdiff(full, pars)
  any_gq_kept <- any(kept %in% gq_only)
  any_tp_kept <- any(kept %in% tp_only)

  list(
    include_tp = any_tp_kept || any_gq_kept,
    include_gq = any_gq_kept
  )
}

#' Constrained parameter names bridgestan returns for the given flags
#'
#' Mirrors `StanModel::with_constrain_flags` in `model.rs`. `(FALSE, TRUE)`
#' is unreachable: `resolve_constrain_flags` enforces `include_tp = TRUE`
#' whenever `include_gq = TRUE`.
#' @noRd
constrain_names_for_flags <- function(handle, include_tp, include_gq) {
  if (include_gq) {
    bs_full_names(handle)
  } else if (include_tp) {
    bs_block_tp_names(handle)
  } else {
    bs_block_names(handle)
  }
}

#' Resolve `pars` / `include` to 0-indexed column indices
#'
#' Operates on bridgestan dot-indexed names so the returned indices line up
#' with the Arrow `value` column ordering — no R-side reshuffling needed.
#' Returns `NULL` for `pars = NULL` (signalling "keep everything") or when
#' nothing in `pars` is in scope (e.g. an `include = FALSE` blacklist that
#' only names columns already skipped via the constrain flags).
#'
#' Validation of unknown names happens upstream in `resolve_constrain_flags`;
#' this function trusts its input.
#' @noRd
resolve_keep_indices <- function(full_names, pars, include) {
  if (is.null(pars)) return(NULL)

  in_scope <- intersect(pars, block_prefixes(full_names))
  if (length(in_scope) == 0L) {
    # Whitelist with no in-scope names = "keep nothing" — surface as an error
    # so a programmatic `pars = intersect(user_pars, available)` that came up
    # empty doesn't silently fall back to keeping everything. Blacklist is
    # benign: the named excludes were already skipped via the constrain flags,
    # so there's nothing left to drop.
    if (include) {
      stop("Parameter selection would remove all variables.", call. = FALSE)
    }
    return(NULL)
  }

  pat <- paste0("^(", paste(escape_regex(in_scope), collapse = "|"), ")(\\.|$)")
  matched <- grepl(pat, full_names)
  keep <- if (include) matched else !matched
  if (!any(keep)) {
    stop("Parameter selection would remove all variables.", call. = FALSE)
  }

  as.integer(which(keep) - 1L)
}

#' Escape special regex characters in a string
#' @noRd
escape_regex <- function(x) {
  gsub("([.\\^$*+?{}()\\[\\]|])", "\\\\\\1", x)
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
