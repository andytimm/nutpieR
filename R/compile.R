#' Compile a Stan model
#'
#' Compiles a Stan model to a shared library using BridgeStan.
#' Downloads BridgeStan sources on first use (this is slow).
#'
#' @param stan_file Path to a `.stan` file.
#' @return An object of class `"nutpie_model"` containing the path to the
#'   compiled shared library.
#' @export
nutpie_compile_model <- function(stan_file) {
  stan_file <- normalizePath(stan_file, mustWork = TRUE)
  lib_path <- compile_stan_model(stan_file)
  structure(
    list(lib_path = lib_path, stan_file = stan_file),
    class = "nutpie_model"
  )
}

#' @export
print.nutpie_model <- function(x, ...) {
  cat("nutpie Stan model\n")
  cat("  Stan file:", x$stan_file, "\n")
  cat("  Library:  ", x$lib_path, "\n")
  invisible(x)
}
