#' Inspect the density evaluator layout
#'
#' Return the exact unconstrained parameter layout passed to a density evaluator at
#' bind time. This view is for evaluator authors: names use BridgeStan's raw dot
#' notation rather than the bracket notation used in R output. See
#' `system.file("examples/byold/README.md", package = "nutpieR")` for the
#' evaluator-writing workflow.
#'
#' @param reference A compiled Stan model from [nutpie_compile_model()].
#' @param data Reference data, serialized as for [nutpie_attach_density_evaluator()]. Use
#'   the same data that will be supplied when attaching the evaluator.
#' @return A list with `ndim`, the raw ordered `names`, and `layout`, the exact
#'   newline-separated ABI string with no trailing newline.
#' @export
nutpie_density_layout <- function(reference, data = NULL) {
  if (inherits(reference, "nutpie_density_evaluator_model")) {
    stop("Supply the original Stan reference, not an existing evaluator binding.", call. = FALSE)
  }
  data_json <- as.character(resolve_data(data))
  if (!nzchar(data_json)) data_json <- "{}"
  handle <- bs_open(resolve_model(reference), data_json, 0L)
  raw_names <- bs_unc_names(handle)
  list(
    ndim = as.integer(bs_ndim_unc(handle)),
    names = raw_names,
    layout = paste(raw_names, collapse = "\n")
  )
}

#' Attach an experimental custom density evaluator
#'
#' Bind a trusted shared library to a Stan reference and a data snapshot, then
#' pass the bound model to [nutpie_sample()]. The evaluator replaces log-density
#' and gradient evaluation. BridgeStan still supplies initialization, transforms,
#' parameter names, transformed parameters and generated quantities. nutpieR
#' still runs the same nuts-rs NUTS sampler.
#'
#' Bind again to change data. Handles work only in the current R session;
#' reattach the evaluator after restoring a serialized object. A bound model keeps
#' the data snapshot supplied at attach time, and `nutpie_sample()` does not
#' accept replacement data for it.
#' Use [nutpie_validate_density_evaluator()] to compare numerical results before sampling.
#' This check is advisory; sampling does not run it automatically.
#'
#' @section Writing a density evaluator:
#' The shared library must implement `nutpier_density_evaluator_v1.h` and satisfy its
#' ownership, buffer, error, and thread-safety contract. Native code runs inside
#' R and can crash or corrupt the process. Density evaluators must match BridgeStan's
#' `propto = TRUE`, `jacobian = TRUE` convention and the reference's ordered
#' unconstrained coordinates. See
#' `system.file("examples/byold/README.md", package = "nutpieR")` for the
#' maintained evaluator-writing workflow.
#'
#' Data uses the same serialization as [nutpie_sample()]: lists pass through
#' `jsonlite::toJSON(auto_unbox = TRUE, digits = NA)`, while JSON strings and
#' files pass through unchanged. With no data, the snapshot is `{}`. Density evaluators
#' must reject missing, nonfinite, or wrongly shaped values they cannot use.
#'
#' @param reference A compiled Stan model from [nutpie_compile_model()].
#' @param library Path to a shared library implementing ABI version 1.
#' @param data Reference and density evaluator data, as for [nutpie_sample()].
#' @return An immutable, session-local `nutpie_density_evaluator_model`.
#' @export
nutpie_attach_density_evaluator <- function(reference, library, data = NULL) {
  if (inherits(reference, "nutpie_density_evaluator_model")) {
    stop("Supply the original Stan reference, not an existing evaluator binding.", call. = FALSE)
  }
  gate_progress_for_tbb("none")
  lib_path <- resolve_model(reference)
  if (!is.character(library) || length(library) != 1L || is.na(library)) {
    stop("`library` must be one shared-library path.", call. = FALSE)
  }
  library <- normalizePath(library, mustWork = TRUE)
  data_json <- as.character(resolve_data(data))
  if (!nzchar(data_json)) data_json <- "{}"
  bs_ptr <- bs_open(lib_path, data_json, 0L)
  gate_progress_for_tbb("none")
  evaluator_ptr <- density_evaluator_bind(bs_ptr, library, data_json)
  bound <- list2env(list(reference = reference, data_json = data_json,
                        bs_ptr = bs_ptr, evaluator_ptr = evaluator_ptr,
                        evaluator_path = library, lib_path = lib_path,
                        unc_names = bs_unc_names(bs_ptr), ndim = bs_ndim_unc(bs_ptr)),
                   parent = emptyenv())
  class(bound) <- c("nutpie_density_evaluator_model", "nutpie_model")
  lockEnvironment(bound, bindings = TRUE)
  bound
}

#' @export
print.nutpie_density_evaluator_model <- function(x, ...) {
  cat("Experimental density evaluator binding\n", x$evaluator_path, "\n",
      x$ndim, " unconstrained parameters; validation is explicit and advisory.\n", sep = "")
  invisible(x)
}
