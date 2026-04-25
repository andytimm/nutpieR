#' Parameter names of a compiled Stan model
#'
#' Returns the parameter names of a compiled Stan model, converted from
#' BridgeStan's dot-indexed form (`"beta.1.2"`) to Stan's bracket convention
#' (`"beta[1,2]"`).
#'
#' @param model A `"nutpie_model"` object from [nutpie_compile_model()].
#' @param data Model data (same format as [nutpie_sample()]'s `data` argument).
#'   Required if the model has a `data` block.
#' @param which One of `"block"`, `"unconstrained"`, `"full"`:
#'   \describe{
#'     \item{`"block"` (default)}{Block-level parameter names — the names you
#'       pass as keys to [nutpie_sample()]'s `init` argument.}
#'     \item{`"unconstrained"`}{Unconstrained internal parameter names.}
#'     \item{`"full"`}{Block parameters plus transformed parameters and
#'       generated quantities (the full output draws space).}
#'   }
#' @param unconstrained Deprecated. If non-`NULL`, overrides `which`:
#'   `TRUE` maps to `"unconstrained"`, `FALSE` maps to `"full"`. Emits a
#'   deprecation warning and will be removed in a future version.
#' @return A character vector of parameter names.
#' @export
nutpie_param_names <- function(model, data = NULL,
                               which = c("block", "unconstrained", "full"),
                               unconstrained = NULL) {
  if (!is.null(unconstrained)) {
    warning(
      "`unconstrained` is deprecated; use `which = \"unconstrained\"` or ",
      "`which = \"full\"` instead. `unconstrained` will be removed in a ",
      "future version.",
      call. = FALSE
    )
    which <- if (isTRUE(unconstrained)) "unconstrained" else "full"
  } else {
    which <- match.arg(which)
  }
  handle <- bs_open(resolve_model(model), resolve_data(data), 0L)
  raw <- switch(which,
    block = bs_block_names(handle),
    unconstrained = bs_unc_names(handle),
    full = bs_full_names(handle)
  )
  dot_to_bracket(raw)
}

#' Map a constrained parameter list to the unconstrained scale
#'
#' Introspection / debugging helper. Takes a named list of parameter values in
#' the constrained (user-facing) space and returns the corresponding
#' unconstrained vector that BridgeStan uses internally. Useful when
#' inspecting how Stan's unconstraining transform maps your values, or when
#' sanity-checking a model's parameter order.
#'
#' For setting sampler starting points, pass constrained values directly to
#' [nutpie_sample()]'s `init` argument (e.g. `init = list(mu = 0, sigma = 1)`)
#' — this function is not part of the normal init workflow.
#'
#' All parameters declared in the `parameters` block must be supplied.
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
