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

  list(positions = NULL, jitter = TRUE)
}

#' Expand a (possibly partial) constrained named list into a full unconstrained
#' vector. Missing parameters are filled by constraining a uniform(-2, 2) draw
#' in unconstrained space, then overlaying the user's values in the constrained
#' JSON representation, and finally mapping the combined dict back to
#' unconstrained space.
#' @noRd
expand_constrained_init <- function(params_list, handle) {
  unc_names_raw <- bs_unc_names(handle)
  valid_block_names <- unique(sub("\\..*$", "", unc_names_raw))

  bad <- setdiff(names(params_list), valid_block_names)
  if (length(bad) > 0L) {
    stop("Unknown parameter(s) in init: ", paste(bad, collapse = ", "),
         "\nAvailable: ", paste(valid_block_names, collapse = ", "),
         call. = FALSE)
  }

  missing_params <- setdiff(valid_block_names, names(params_list))
  structured <- if (length(missing_params) == 0L) {
    params_list
  } else {
    # Fill missing params by constraining a random unconstrained draw, so we
    # get well-formed shapes (arrays / matrices) for every declared parameter.
    rand_unc <- stats::runif(length(unc_names_raw), min = -2, max = 2)
    flat_con <- bs_param_constrain(
      handle, rand_unc,
      as.integer(sample.int(.Machine$integer.max, 1L))
    )
    con_names_raw <- bs_full_names(handle)
    defaults <- reconstruct_stan_json(flat_con, con_names_raw)
    defaults <- defaults[intersect(names(defaults), valid_block_names)]
    modifyList(defaults, params_list)
  }

  init_json <- jsonlite::toJSON(structured, auto_unbox = TRUE, digits = NA,
                                matrix = "columnmajor")
  bs_param_unconstrain_json(handle, init_json)
}

#' Reconstruct Stan-style nested list from a flat vector + dot-indexed names.
#' Given `flat_values = c(1, 2, 3, 4)` and
#' `indexed_names = c("M.1.1", "M.2.1", "M.1.2", "M.2.2")` returns
#' `list(M = matrix(c(1,2,3,4), 2, 2))` (column-major).
#' Scalars and 1D arrays are handled too.
#' @noRd
reconstruct_stan_json <- function(flat_values, indexed_names) {
  if (length(flat_values) != length(indexed_names)) {
    stop("flat_values and indexed_names must have the same length.", call. = FALSE)
  }
  if (length(flat_values) == 0L) return(list())

  # Parse each name into base and integer index vector.
  parsed <- lapply(indexed_names, function(nm) {
    parts <- strsplit(nm, ".", fixed = TRUE)[[1L]]
    base <- parts[1L]
    idx <- if (length(parts) > 1L) as.integer(parts[-1L]) else integer(0)
    list(base = base, idx = idx)
  })
  bases <- vapply(parsed, `[[`, character(1), "base")

  out <- list()
  for (b in unique(bases)) {
    mask <- bases == b
    entries <- parsed[mask]
    vals <- flat_values[mask]

    rank <- length(entries[[1L]]$idx)
    if (rank == 0L) {
      # Scalar; there should be only one entry.
      out[[b]] <- vals[[1L]]
      next
    }

    # Check consistent rank.
    ranks <- vapply(entries, function(e) length(e$idx), integer(1))
    if (!all(ranks == rank)) {
      stop("Inconsistent index rank for parameter '", b, "'.", call. = FALSE)
    }

    # Determine dims.
    idx_matrix <- do.call(rbind, lapply(entries, `[[`, "idx"))
    dims <- apply(idx_matrix, 2, max)

    if (prod(dims) != length(vals)) {
      stop("Expected ", prod(dims), " values for '", b, "' (dims = ",
           paste(dims, collapse = "x"), ") but got ", length(vals), ".",
           call. = FALSE)
    }

    # BridgeStan emits column-major (last-index-major) order: for a matrix
    # M[i,j], index i varies fastest. That is exactly R's array storage order
    # when dimensions are listed in declaration order, so we can just assign.
    arr <- array(vals, dim = dims)
    out[[b]] <- arr
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
