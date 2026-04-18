#' Parameter names of a compiled Stan model
#'
#' Returns the parameter names of a compiled Stan model, converted from
#' BridgeStan's dot-indexed form (`"beta.1.2"`) to Stan's bracket convention
#' (`"beta[1,2]"`).
#'
#' @param model A `"nutpie_model"` object from [nutpie_compile_model()].
#' @param data Model data (same format as [nutpie_sample()]'s `data` argument).
#'   Required if the model has a `data` block.
#' @param unconstrained If `TRUE` (default), return unconstrained parameter
#'   names (the space used by `init_unconstrained`). If `FALSE`, return the
#'   full constrained names including transformed parameters and generated
#'   quantities (the space users usually think in).
#' @return A character vector of parameter names.
#' @export
nutpie_param_names <- function(model, data = NULL, unconstrained = TRUE) {
  handle <- bs_open(resolve_model(model), resolve_data(data), 0L)
  raw <- if (isTRUE(unconstrained)) bs_unc_names(handle) else bs_full_names(handle)
  dot_to_bracket(raw)
}

#' Map a constrained parameter list to the unconstrained scale
#'
#' Takes a named list of parameter values in the constrained (user-facing)
#' space, and returns the corresponding unconstrained vector that BridgeStan
#' uses internally. Useful for building `init_unconstrained` from familiar
#' parameter values.
#'
#' All parameters declared in the `parameters` block must be supplied. For
#' partial inits, use [nutpie_sample()]'s `init` argument instead.
#'
#' @param model A `"nutpie_model"` object.
#' @param params A named list of parameter values. Names must match the
#'   parameter names in the Stan program. Values may be scalars, vectors,
#'   matrices, or arrays — matching the declared shape.
#' @param data Model data (same format as [nutpie_sample()]'s `data`).
#'   Required if the model has a `data` block.
#' @return A named numeric vector whose names are the unconstrained parameter
#'   names (in BridgeStan's internal order).
#' @export
nutpie_unconstrain <- function(model, params, data = NULL) {
  if (!is.list(params) || is.null(names(params)) || any(names(params) == "")) {
    stop("`params` must be a fully named list.", call. = FALSE)
  }
  handle <- bs_open(resolve_model(model), resolve_data(data), 0L)
  theta_flat <- flat_overlay(params, bs_block_names(handle))
  unc <- bs_param_unconstrain(handle, theta_flat)
  stats::setNames(unc, dot_to_bracket(bs_unc_names(handle)))
}
