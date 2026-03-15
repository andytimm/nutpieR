#' Compile a Stan model
#'
#' Compiles a Stan model to a shared library using BridgeStan.
#' Downloads BridgeStan sources on first use (this is slow).
#'
#' @param stan_file Path to a `.stan` file. Exactly one of `stan_file` or
#'   `code` must be provided.
#' @param code A string containing Stan model code. If provided, the code is
#'   written to a temporary `.stan` file and compiled.
#' @param stanc_args Character vector of extra arguments passed to the
#'   `stanc` compiler (e.g., `"--O1"` for optimization).
#' @param compile_args Character vector of extra arguments passed to `make`
#'   during compilation.
#' @return An object of class `"nutpie_model"` containing the path to the
#'   compiled shared library.
#' @export
nutpie_compile_model <- function(stan_file = NULL, code = NULL,
                                 stanc_args = character(),
                                 compile_args = character()) {
  if (!is.null(stan_file) && !is.null(code)) {
    stop("Provide exactly one of `stan_file` or `code`, not both.", call. = FALSE)
  }
  if (is.null(stan_file) && is.null(code)) {
    stop("Provide exactly one of `stan_file` or `code`.", call. = FALSE)
  }

  if (!is.null(code)) {
    stan_file <- tempfile(fileext = ".stan")
    writeLines(code, stan_file)
  }

  stan_file <- normalizePath(stan_file, mustWork = TRUE)
  lib_path <- compile_stan_model(stan_file, as.character(stanc_args),
                                  as.character(compile_args))
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
