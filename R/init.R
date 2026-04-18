#' Resolve the three init-related arguments into sampler-ready positions.
#'
#' Returns `list(positions = NULL | list of numeric vectors, jitter = logical)`.
#' - `positions = NULL` means "use sampler default" (Uniform(-2, 2) per chain).
#' - Otherwise, `positions` has length 1 (broadcast) or `num_chains`, and each
#'   inner vector is an unconstrained parameter vector (length `ndim_unc`).
#' Mutual exclusion between `init`, `init_unconstrained`, `init_mean` is
#' enforced here. `init_mean` is returned as a *jittered* init_positions of
#' length 1 (legacy behaviour).
#' @noRd
resolve_init <- function(init, init_unconstrained, init_mean, handle,
                          num_chains) {
  supplied <- c(!is.null(init), !is.null(init_unconstrained), !is.null(init_mean))
  if (sum(supplied) > 1L) {
    stop(
      "At most one of `init`, `init_unconstrained`, `init_mean` may be supplied.",
      call. = FALSE
    )
  }

  if (!is.null(init)) {
    positions <- init_list_to_positions(init, handle, num_chains)
    return(list(positions = positions, jitter = FALSE))
  }

  if (!is.null(init_unconstrained)) {
    positions <- init_unc_to_positions(init_unconstrained, handle, num_chains)
    return(list(positions = positions, jitter = FALSE))
  }

  if (!is.null(init_mean)) {
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

  list(positions = NULL, jitter = TRUE)
}

#' Expand a (possibly partial) constrained named list into a full unconstrained
#' vector. Missing parameters are filled by constraining a uniform(-2, 2) draw
#' in unconstrained space to produce well-shaped defaults for each declared
#' block-level parameter.
#' @noRd
expand_constrained_init <- function(params_list, handle) {
  block_names <- bs_block_names(handle)
  valid_bases <- unique(sub("\\..*$", "", block_names))

  bad <- setdiff(names(params_list), valid_bases)
  if (length(bad) > 0L) {
    stop("Unknown parameter(s) in init: ", paste(bad, collapse = ", "),
         "\nAvailable: ", paste(valid_bases, collapse = ", "),
         call. = FALSE)
  }

  missing_params <- setdiff(valid_bases, names(params_list))
  defaults <- if (length(missing_params) == 0L) {
    NULL
  } else {
    rand_unc <- stats::runif(bs_ndim_unc(handle), min = -2, max = 2)
    full_con <- bs_param_constrain(
      handle, rand_unc,
      as.integer(sample.int(.Machine$integer.max, 1L))
    )
    full_con[seq_along(block_names)]
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

  for (base in user_names) {
    idx <- which(bases == base)
    flat <- as.numeric(user_list[[base]])
    if (length(flat) != length(idx)) {
      stop("Parameter '", base, "' has ", length(flat),
           " values but expected ", length(idx), ".", call. = FALSE)
    }
    out[idx] <- flat
  }
  out
}

#' Convert an `init` argument (single list | list-of-lists | path | list of
#' paths) to a list of unconstrained position vectors.
#' @noRd
init_list_to_positions <- function(init, handle, num_chains) {
  # Character path(s): read JSON and recurse.
  if (is.character(init)) {
    if (length(init) == 1L) {
      parsed <- jsonlite::fromJSON(init, simplifyVector = TRUE)
      return(init_list_to_positions(parsed, handle, num_chains))
    }
    if (length(init) == num_chains) {
      parsed <- lapply(init, jsonlite::fromJSON, simplifyVector = TRUE)
      return(init_list_to_positions(parsed, handle, num_chains))
    }
    stop("When `init` is a character vector, its length must be 1 or num_chains.",
         call. = FALSE)
  }

  if (!is.list(init)) {
    stop("`init` must be a named list, a list of named lists, or a JSON path.",
         call. = FALSE)
  }

  # Detect list-of-lists vs single named list.
  if (is_single_named_param_list(init)) {
    pos <- expand_constrained_init(init, handle)
    return(list(pos))  # broadcast
  }

  # List of per-chain inits.
  if (length(init) != num_chains) {
    stop("`init` must have length ", num_chains, " (num_chains), got ",
         length(init), ".", call. = FALSE)
  }
  lapply(init, function(p) {
    if (is.character(p)) p <- jsonlite::fromJSON(p, simplifyVector = TRUE)
    if (!is.list(p) || !is_single_named_param_list(p)) {
      stop("Each per-chain `init` element must be a named list of parameter values.",
           call. = FALSE)
    }
    expand_constrained_init(p, handle)
  })
}

#' Heuristic: a list is a "single param list" if it is named and no element is
#' itself a list-of-lists. (Users should not declare top-level Stan params that
#' are lists.)
#' @noRd
is_single_named_param_list <- function(x) {
  if (!is.list(x)) return(FALSE)
  nms <- names(x)
  if (is.null(nms) || any(nms == "")) return(FALSE)
  # If any element is itself a named list, assume it's a per-chain wrapper.
  any_nested_named_list <- any(vapply(x, function(el) {
    is.list(el) && !is.null(names(el)) && all(names(el) != "")
  }, logical(1)))
  !any_nested_named_list
}

#' Convert `init_unconstrained` argument (named vector or list of them) to a
#' list of unconstrained positions (reordered to BridgeStan's internal order).
#' @noRd
init_unc_to_positions <- function(init_unconstrained, handle, num_chains) {
  unc_names <- dot_to_bracket(bs_unc_names(handle))
  ndim <- length(unc_names)

  if (is.numeric(init_unconstrained)) {
    return(list(reorder_unc_vec(init_unconstrained, unc_names, ndim)))
  }
  if (is.list(init_unconstrained)) {
    if (length(init_unconstrained) != num_chains) {
      stop("`init_unconstrained` list must have length num_chains (", num_chains,
           ").", call. = FALSE)
    }
    return(lapply(init_unconstrained, function(v) {
      if (!is.numeric(v)) {
        stop("Each element of `init_unconstrained` must be a numeric vector.",
             call. = FALSE)
      }
      reorder_unc_vec(v, unc_names, ndim)
    }))
  }
  stop("`init_unconstrained` must be a named numeric vector, or a list of such.",
       call. = FALSE)
}

#' Validate and reorder a named unconstrained vector to BridgeStan's order.
#' @noRd
reorder_unc_vec <- function(v, unc_names, ndim) {
  if (is.null(names(v)) || any(names(v) == "")) {
    stop("`init_unconstrained` must be a named numeric vector (one entry per ",
         "unconstrained parameter). Use nutpie_param_names(model, ",
         "unconstrained = TRUE) to see the expected names.",
         call. = FALSE)
  }
  if (length(v) != ndim) {
    stop("`init_unconstrained` has length ", length(v), " but model has ",
         ndim, " unconstrained parameters.", call. = FALSE)
  }
  missing <- setdiff(unc_names, names(v))
  extra <- setdiff(names(v), unc_names)
  if (length(missing) > 0L || length(extra) > 0L) {
    msg <- "Names of `init_unconstrained` do not match model parameters."
    if (length(missing) > 0L) {
      msg <- paste0(msg, "\n  Missing: ", paste(missing, collapse = ", "))
    }
    if (length(extra) > 0L) {
      msg <- paste0(msg, "\n  Unknown: ", paste(extra, collapse = ", "))
    }
    stop(msg, call. = FALSE)
  }
  as.numeric(v[unc_names])
}
