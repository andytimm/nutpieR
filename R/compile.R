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
#' @param verbose Integer controlling compilation output. `0` = silent,
#'   `1` (default) = print status messages.
#'   Note: full make/stanc output (verbose=2) is not yet supported because
#'   bridgestan captures subprocess output internally rather than streaming it.
#' @return An object of class `"nutpie_model"` containing the path to the
#'   compiled shared library.
#' @export
nutpie_compile_model <- function(stan_file = NULL, code = NULL,
                                 stanc_args = character(),
                                 compile_args = character(),
                                 verbose = 1L) {
  if (!is.null(stan_file) && !is.null(code)) {
    stop("Provide exactly one of `stan_file` or `code`, not both.", call. = FALSE)
  }
  if (is.null(stan_file) && is.null(code)) {
    stop("Provide exactly one of `stan_file` or `code`.", call. = FALSE)
  }

  # bridgestan::compile_model shells out to make

  if (Sys.which("make") == "") {
    platform_hint <- if (.Platform$OS.type == "windows") {
      "Install Rtools (https://cran.r-project.org/bin/windows/Rtools/) and ensure it is on your PATH."
    } else if (Sys.info()[["sysname"]] == "Darwin") {
      "Install Xcode Command Line Tools: xcode-select --install"
    } else {
      "Install build-essential: sudo apt install build-essential (Debian/Ubuntu) or sudo dnf install make gcc-c++ (Fedora)"
    }
    stop(
      "`make` is required to compile Stan models but was not found on PATH.\n",
      platform_hint,
      call. = FALSE
    )
  }

  if (!is.null(code)) {
    stan_file <- tempfile(fileext = ".stan")
    writeLines(code, stan_file)
  }

  stan_file <- normalizePath(stan_file, mustWork = TRUE)

  # bridgestan shells out to make, which can't handle spaces in paths.
  # Copy the stan file to a temp dir if the path contains spaces.
  if (grepl(" ", stan_file)) {
    tmp_dir <- tempfile("nutpie_model_")
    dir.create(tmp_dir)
    tmp_stan <- file.path(tmp_dir, basename(stan_file))
    file.copy(stan_file, tmp_stan)
    stan_file <- tmp_stan
  }
  verbose <- as.integer(verbose)
  if (verbose >= 1L) {
    message("Compiling Stan model...")
    start_time <- proc.time()[["elapsed"]]
  }
  lib_path <- compile_stan_model(stan_file, as.character(stanc_args),
                                  as.character(compile_args))
  if (verbose >= 1L) {
    elapsed <- proc.time()[["elapsed"]] - start_time
    message(sprintf("Compiled in %.1fs", elapsed))
  }
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
