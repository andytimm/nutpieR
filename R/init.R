#' Evaluate `expr` under a locally-set RNG seed, restoring the caller's global
#' RNG state on exit. Lets functions that need randomness be driven by an
#' explicit seed without leaking any RNG advancement back to the caller.
#' @noRd
with_local_seed <- function(seed, expr) {
  old_exists <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (old_exists) {
    old <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  }
  on.exit({
    if (old_exists) {
      assign(".Random.seed", old, envir = globalenv())
    } else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }
  }, add = TRUE)
  set.seed(seed)
  expr
}

#' Resolve the `init` argument into sampler-ready positions.
#'
#' Returns `list(positions = NULL | list of numeric vectors, jitter = logical)`.
#' - `positions = NULL` means "use sampler default" (Uniform(-2, 2) per chain).
#' - Otherwise, `positions` has length 1 (broadcast) or `num_chains`, and each
#'   inner vector is an unconstrained parameter vector (length `ndim_unc`).
#' `init_mean` is kept as a soft-deprecated alias — when supplied it emits a
#' warning and is routed through the numeric-vector-on-unconstrained-space
#' path with jitter. `seed` drives reproducibility of any random fill used by
#' partial constrained inits and numeric-scalar Uniform(-x, x) draws.
#' @noRd
resolve_init <- function(init, init_mean, handle, num_chains, seed) {
  if (!is.null(init_mean)) {
    if (!is.null(init)) {
      stop("Supply either `init` or `init_mean`, not both.", call. = FALSE)
    }
    warning(
      "`init_mean` is deprecated; use `init = function(chain_id) ...`, ",
      "`init = <scalar>` for Uniform(-x, x), or `init = <named list>`. ",
      "`init_mean` will be removed in a future version.",
      call. = FALSE
    )
    if (!is.numeric(init_mean)) {
      stop("`init_mean` must be a numeric vector.", call. = FALSE)
    }
    ndim <- bs_ndim_unc(handle)
    vec <- if (length(init_mean) == 1L) {
      rep(as.numeric(init_mean), ndim)
    } else {
      as.numeric(init_mean)
    }
    if (length(vec) != ndim) {
      stop("`init_mean` length (", length(vec),
           ") does not match model dimension (", ndim, ").", call. = FALSE)
    }
    return(list(positions = list(vec), jitter = TRUE))
  }

  if (is.null(init)) {
    return(list(positions = NULL, jitter = TRUE))
  }

  if (is.function(init)) {
    positions <- resolve_function_init(init, handle, num_chains, seed)
    return(list(positions = positions, jitter = FALSE))
  }

  if (is.numeric(init) && !is.list(init) && length(init) == 1L) {
    positions <- resolve_numeric_init(init, handle, num_chains, seed)
    return(list(positions = positions, jitter = FALSE))
  }

  if (is.character(init)) {
    if (length(init) == 1L) {
      parsed <- jsonlite::fromJSON(init, simplifyVector = TRUE)
      return(resolve_init(parsed, NULL, handle, num_chains, seed))
    }
    if (length(init) == num_chains) {
      parsed <- lapply(init, jsonlite::fromJSON, simplifyVector = TRUE)
      return(resolve_init(parsed, NULL, handle, num_chains, seed))
    }
    stop("When `init` is a character vector, its length must be 1 or num_chains.",
         call. = FALSE)
  }

  if (is.list(init)) {
    positions <- init_list_to_positions(init, handle, num_chains, seed)
    return(list(positions = positions, jitter = FALSE))
  }

  stop("`init` must be NULL, a scalar numeric, a function, a named list, a ",
       "list of named lists, or a JSON path.", call. = FALSE)
}

#' Resolve a scalar-numeric `init`: Uniform(-x, x) per chain on the
#' unconstrained scale. `x == 0` means every chain starts at the origin (no
#' jitter). For `x > 0`, per-chain seeds are derived deterministically from
#' `seed` so the same `seed` gives the same starts.
#' @noRd
resolve_numeric_init <- function(x, handle, num_chains, seed) {
  x <- as.numeric(x)
  if (!is.finite(x) || x < 0) {
    stop("`init` scalar must be a non-negative finite number (got ", x, ").",
         call. = FALSE)
  }
  ndim <- bs_ndim_unc(handle)
  if (x == 0) {
    return(list(rep(0, ndim)))
  }
  with_local_seed(seed, {
    lapply(seq_len(num_chains), function(i) {
      stats::runif(ndim, min = -x, max = x)
    })
  })
}

#' Resolve a function-form `init`: call `f(chain_id)` for `chain_id` in
#' `1:num_chains` and treat each result as a (possibly partial) constrained
#' named list. Each per-chain list is expanded via `expand_constrained_init`
#' with its own deterministic seed.
#' @noRd
resolve_function_init <- function(f, handle, num_chains, seed) {
  if (length(formals(f)) < 1L) {
    stop("`init` function must accept at least one argument (chain_id).",
         call. = FALSE)
  }
  block_names <- bs_block_names(handle)
  valid_bases <- unique(sub("\\..*$", "", block_names))
  chain_seeds <- with_local_seed(seed, {
    sample.int(.Machine$integer.max, num_chains)
  })
  lapply(seq_len(num_chains), function(i) {
    val <- f(i)
    if (!is.list(val) || is.null(names(val)) || any(names(val) == "")) {
      stop("`init` function must return a named list of parameter values ",
           "(got result for chain_id = ", i, " that is not a named list).",
           call. = FALSE)
    }
    if (!all(vapply(val, is.numeric, logical(1)))) {
      stop("`init` function returned non-numeric values for chain_id = ", i,
           ".", call. = FALSE)
    }
    expand_constrained_init(val, handle, block_names, valid_bases,
                            seed = chain_seeds[i])
  })
}

#' Expand a (possibly partial) constrained named list into a full unconstrained
#' vector. Missing parameters are filled by constraining a uniform(-2, 2) draw
#' in unconstrained space to produce well-shaped defaults for each declared
#' block-level parameter. `block_names` / `valid_bases` may be supplied by
#' callers that hoist them out of a per-chain loop. When `seed` is non-NULL the
#' random fill is driven by that seed (global RNG state is preserved).
#' @noRd
expand_constrained_init <- function(params_list, handle,
                                    block_names = bs_block_names(handle),
                                    valid_bases = unique(sub("\\..*$", "", block_names)),
                                    seed = NULL) {
  defaults <- if (all(valid_bases %in% names(params_list))) {
    NULL
  } else {
    fill <- function() {
      rand_unc <- stats::runif(bs_ndim_unc(handle), min = -2, max = 2)
      full_con <- bs_param_constrain(
        handle, rand_unc,
        as.integer(sample.int(.Machine$integer.max, 1L))
      )
      full_con[seq_along(block_names)]
    }
    if (is.null(seed)) fill() else with_local_seed(seed, fill())
  }

  theta_flat <- flat_overlay(params_list, block_names, defaults)
  bs_param_unconstrain(handle, theta_flat)
}

#' Overlay a user-provided named list onto the flat block-level parameter
#' vector that BridgeStan's `param_unconstrain` expects.
#'
#' `block_names` is the dot-indexed output of `bs_block_names(handle)`, in
#' BridgeStan's column-major / last-index-major order. Each entry of
#' `user_list` is flattened in R's native (column-major) order — which matches
#' BridgeStan's order for any rank — and placed at the corresponding indices.
#' When `defaults` is NULL, every base name in `block_names` must appear in
#' `user_list`; otherwise missing entries come from `defaults`.
#' @noRd
flat_overlay <- function(user_list, block_names, defaults = NULL) {
  if (length(block_names) == 0L) return(numeric(0))

  bases <- sub("\\..*$", "", block_names)
  unique_bases <- unique(bases)

  user_names <- names(user_list)
  bad <- setdiff(user_names, unique_bases)
  if (length(bad) > 0L) {
    stop("Unknown parameter(s): ", paste(bad, collapse = ", "),
         "\nAvailable: ", paste(unique_bases, collapse = ", "),
         call. = FALSE)
  }

  if (is.null(defaults)) {
    missing_nms <- setdiff(unique_bases, user_names)
    if (length(missing_nms) > 0L) {
      stop("Missing required parameter(s): ",
           paste(missing_nms, collapse = ", "), call. = FALSE)
    }
    out <- numeric(length(block_names))
  } else {
    if (length(defaults) != length(block_names)) {
      stop("defaults length (", length(defaults),
           ") does not match block_names length (",
           length(block_names), ").", call. = FALSE)
    }
    out <- as.numeric(defaults)
  }

  idx_by_base <- split(seq_along(bases), bases)
  for (base in user_names) {
    idx <- idx_by_base[[base]]
    flat <- as.numeric(user_list[[base]])
    if (length(flat) != length(idx)) {
      stop("Parameter '", base, "' has ", length(flat),
           " values but expected ", length(idx), ".", call. = FALSE)
    }
    out[idx] <- flat
  }
  out
}

#' Convert a list-form `init` (single list | list-of-lists) to a list of
#' unconstrained position vectors. `seed` drives reproducibility of any
#' per-chain random fill for partial inits.
#' @noRd
init_list_to_positions <- function(init, handle, num_chains, seed) {
  if (!is.list(init)) {
    stop("`init` must be a named list, a list of named lists, or a JSON path.",
         call. = FALSE)
  }

  block_names <- bs_block_names(handle)
  valid_bases <- unique(sub("\\..*$", "", block_names))

  if (is_single_named_param_list(init)) {
    pos <- expand_constrained_init(init, handle, block_names, valid_bases,
                                   seed = seed)
    return(list(pos))
  }

  if (length(init) != num_chains) {
    stop("`init` must have length ", num_chains, " (num_chains), got ",
         length(init), ".", call. = FALSE)
  }
  chain_seeds <- with_local_seed(seed, {
    sample.int(.Machine$integer.max, num_chains)
  })
  Map(function(p, s) {
    if (is.character(p)) p <- jsonlite::fromJSON(p, simplifyVector = TRUE)
    if (!is.list(p) || !is_single_named_param_list(p)) {
      stop("Each per-chain `init` element must be a named list of parameter values.",
           call. = FALSE)
    }
    expand_constrained_init(p, handle, block_names, valid_bases, seed = s)
  }, init, chain_seeds)
}

#' Heuristic: a list is a "single param list" if it is named and no element is
#' itself a list-of-lists. (Users should not declare top-level Stan params that
#' are lists.)
#' @noRd
is_single_named_param_list <- function(x) {
  if (!is.list(x)) return(FALSE)
  nms <- names(x)
  if (is.null(nms) || any(nms == "")) return(FALSE)
  any_nested_named_list <- any(vapply(x, function(el) {
    is.list(el) && !is.null(names(el)) && all(names(el) != "")
  }, logical(1)))
  !any_nested_named_list
}
