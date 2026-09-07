#' Attach an experimental native density kernel
#'
#' Bind a trusted shared library to a Stan reference and a data snapshot.
#' The kernel replaces only log density and gradient evaluation. BridgeStan
#' still supplies initialization, transforms, parameter names, TP and GQ.
#' Kernels must use `propto = TRUE`, `jacobian = TRUE` and the reference's
#' ordered unconstrained coordinates. Numerical validation is explicit and
#' advisory; sampling does not run it automatically.
#'
#' Native code runs inside R. It must implement `nutpier_kernel_v1.h`, contain
#' its own exceptions, and satisfy its memory and thread-safety contract.
#' Invalid native code can crash or corrupt R. Handles are session-local;
#' serialized objects must be rebound. Bind again to change the data.
#'
#' Data uses the same serialization as [nutpie_sample()]: lists pass through
#' `jsonlite::toJSON(auto_unbox = TRUE, digits = NA)`. Length-one atomic
#' vectors become scalars, arrays retain dimensions, and missing/nonfinite
#' numbers use jsonlite's default JSON strings. Kernels must reject missing,
#' nonfinite or wrongly shaped values they cannot use. JSON strings/files
#' pass through without canonicalization. Fixed-data kernels must compare
#' meaningful values, not JSON byte strings. With no data, the snapshot is `{}`.
#'
#' @param reference A compiled Stan model from [nutpie_compile_model()].
#' @param library Path to a shared library implementing ABI version 1.
#' @param data Reference and kernel data, as for [nutpie_sample()].
#' @return An immutable, session-local `nutpie_kernel_model`.
#' @export
nutpie_attach_kernel <- function(reference, library, data = NULL) {
  if (inherits(reference, "nutpie_kernel_model")) {
    stop("Supply the original Stan reference, not an existing kernel binding.", call. = FALSE)
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
  kernel_ptr <- kernel_bind(bs_ptr, library, data_json)
  bound <- list2env(list(reference = reference, data_json = data_json,
                        bs_ptr = bs_ptr, kernel_ptr = kernel_ptr,
                        kernel_path = library, lib_path = lib_path,
                        unc_names = bs_unc_names(bs_ptr), ndim = bs_ndim_unc(bs_ptr)),
                   parent = emptyenv())
  class(bound) <- c("nutpie_kernel_model", "nutpie_model")
  lockEnvironment(bound, bindings = TRUE)
  bound
}

#' @export
print.nutpie_kernel_model <- function(x, ...) {
  cat("Experimental native kernel binding\n", x$kernel_path, "\n",
      x$ndim, " unconstrained parameters; validation is explicit and advisory.\n", sep = "")
  invisible(x)
}
