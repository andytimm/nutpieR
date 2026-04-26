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

#' Resolve `pars` / `include` to bridgestan `(include_tp, include_gq)` flags
#'
#' When the user has filtered TP / GQ out of the output, we tell bridgestan
#' to skip materializing them in `param_constrain` — saving the per-draw
#' allocation and the Stan-side compute (e.g. the `*_rng` calls in GQ).
#'
#' Resolution rules:
#' - `pars = NULL` → `(TRUE, TRUE)` (full backward compatibility).
#' - Otherwise, classify each block-level prefix as block / TP-only / GQ-only
#'   using `bs_block_names`, `bs_block_tp_names`, `bs_full_names`. Compute the
#'   set of *kept* prefixes (whitelist or blacklist), then:
#'   - `include_gq = TRUE` iff any kept prefix is GQ-only.
#'   - `include_tp = TRUE` iff any kept prefix is TP-only OR `include_gq = TRUE`
#'     (GQ may reference TP, so we keep TP whenever GQ is kept).
#'
#' @param handle An open `BSHandle` external pointer.
#' @param pars Character vector of block-level prefixes, or `NULL`.
#' @param include `TRUE` for whitelist, `FALSE` for blacklist.
#' @return A named list with logical scalars `include_tp`, `include_gq`.
#' @noRd
resolve_constrain_flags <- function(handle, pars, include) {
  if (is.null(pars)) return(list(include_tp = TRUE, include_gq = TRUE))

  block_prefixes <- unique(sub("\\..*", "", bs_block_names(handle)))
  block_tp_prefixes <- unique(sub("\\..*", "", bs_block_tp_names(handle)))
  full_prefixes <- unique(sub("\\..*", "", bs_full_names(handle)))

  bad <- setdiff(pars, full_prefixes)
  if (length(bad) > 0) {
    stop("Unknown parameter(s): ", paste(bad, collapse = ", "),
         "\nAvailable: ", paste(full_prefixes, collapse = ", "),
         call. = FALSE)
  }

  tp_only <- setdiff(block_tp_prefixes, block_prefixes)
  gq_only <- setdiff(full_prefixes, block_tp_prefixes)

  kept <- if (include) pars else setdiff(full_prefixes, pars)
  any_gq_kept <- any(kept %in% gq_only)
  any_tp_kept <- any(kept %in% tp_only)

  list(
    include_tp = any_tp_kept || any_gq_kept,
    include_gq = any_gq_kept
  )
}

#' Constrained parameter names that bridgestan will return for the given flags
#'
#' Mirrors what bridgestan's `param_constrain` writes into its output buffer:
#' block only / block+TP / block+TP+GQ. The `(FALSE, TRUE)` combination is
#' rejected because GQ may reference TP — `resolve_constrain_flags` already
#' enforces `include_tp = TRUE` whenever `include_gq = TRUE`.
#'
#' @noRd
constrain_names_for_flags <- function(handle, include_tp, include_gq) {
  if (include_tp && include_gq) {
    bs_full_names(handle)
  } else if (include_tp) {
    bs_block_tp_names(handle)
  } else if (!include_gq) {
    bs_block_names(handle)
  } else {
    stop("Internal error: include_gq = TRUE requires include_tp = TRUE.",
         call. = FALSE)
  }
}

#' Resolve `pars` / `include` to 0-indexed column indices
#'
#' Operates on bridgestan dot-indexed full names ("beta.1.2"), so the
#' returned indices line up with the Arrow `value` column ordering — no
#' R-side reshuffling needed. Returns `NULL` when no filtering is
#' requested, signalling "keep everything" to the Rust side.
#'
#' @param full_names Character vector of bridgestan dot-indexed names.
#' @param pars Character vector of block-level parameter names, or `NULL`.
#' @param include Logical; if `TRUE`, `pars` is a whitelist; if `FALSE`,
#'   a blacklist.
#' @return 0-indexed integer vector of columns to keep, or `NULL`.
#' @noRd
resolve_keep_indices <- function(full_names, pars, include) {
  if (is.null(pars)) return(NULL)

  block_names <- unique(sub("\\..*", "", full_names))
  bad <- setdiff(pars, block_names)
  if (length(bad) > 0) {
    stop("Unknown parameter(s): ", paste(bad, collapse = ", "),
         "\nAvailable: ", paste(block_names, collapse = ", "),
         call. = FALSE)
  }

  # Match "par" exactly (scalar) or "par.<digits>..." (indexed).
  pat <- paste0("^(", paste(escape_regex(pars), collapse = "|"), ")(\\.|$)")
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
