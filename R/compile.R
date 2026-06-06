#' Compile a Stan model
#'
#' Compiles a Stan model to a shared library using BridgeStan.
#' Downloads BridgeStan sources on first use (this is slow).
#'
#' @section Caching:
#'
#' Compiled artifacts are stored in a content-hashed cache under
#' [`nutpie_cache_dir()`][nutpie_cache_dir] (one subdirectory per unique
#' source + flags + BridgeStan version), regardless of whether the model
#' was passed as `stan_file = ...` or `code = "..."`. A subsequent call
#' with identical inputs is a near-instant cache hit.
#'
#' For `stan_file = ...`, the transitive `#include` set is hashed
#' together with the main file, so editing an included file (or the main
#' file itself) busts the cache and triggers a recompile.
#'
#' The cache is bounded by [`nutpie_prune_cache()`][nutpie_prune_cache],
#' which runs automatically at the end of every successful compile
#' (cap: 16 entries, min age before eviction: 14 days).
#'
#' Cache controls:
#'
#' * `cache = FALSE` on a single call --- compile to a fresh tempdir for
#'   this call only, without touching the persistent cache.
#' * `Sys.setenv(NUTPIER_DISABLE_COMPILE_CACHE = "1")` --- same effect
#'   process-wide.
#' * [`nutpie_clear_cache()`][nutpie_clear_cache] wipes the cache.
#'
#' @section Note on storage location:
#'
#' Prior nutpieR versions wrote `<basename>_model.so` *next to* the
#' source `.stan` file, matching cmdstanr's convention. nutpieR now uses
#' a content-hashed cache directory instead. This change is required for
#' correctness: when the same `.stan` path is reloaded after a recompile,
#' the OS dynamic linker (`dlopen`) returns the previously loaded library
#' rather than the new one, so edits silently had no effect (see GitHub
#' issue #23). Distinct content → distinct path → fresh `dlopen`. Any
#' stale `<basename>_model.so` and `<basename>_model.cache_meta` files
#' left over from earlier versions can be deleted; nutpieR no longer
#' reads or writes them.
#'
#' @param stan_file Path to a `.stan` file. Exactly one of `stan_file` or
#'   `code` must be provided.
#' @param code A string containing Stan model code.
#' @param stanc_args Character vector of extra arguments passed to the
#'   `stanc` compiler (e.g., `"--O1"` for optimization).
#' @param compile_args Character vector of extra arguments passed to `make`
#'   during compilation.
#' @param verbose Integer controlling compilation output. `0` = silent,
#'   `1` (default) = print status messages.
#'   Note: full make/stanc output (verbose=2) is not yet supported because
#'   bridgestan captures subprocess output internally rather than streaming it.
#' @param cache Logical, default `TRUE`. When `TRUE`, reuse a previously
#'   compiled artifact when the source, BridgeStan version, and compile
#'   flags all match. When `FALSE`, compile to a fresh tempdir without
#'   touching the persistent cache.
#' @return An object of class `"nutpie_model"` containing the path to the
#'   compiled shared library.
#' @examples
#' \dontrun{
#' # From a .stan file
#' model <- nutpie_compile_model(stan_file = "my_model.stan")
#'
#' # From an inline code string
#' model <- nutpie_compile_model(code = "
#'   data { int<lower=0> N; array[N] int<lower=0,upper=1> y; }
#'   parameters { real<lower=0,upper=1> theta; }
#'   model { theta ~ beta(1, 1); y ~ bernoulli(theta); }
#' ")
#' }
#' @export
nutpie_compile_model <- function(stan_file = NULL, code = NULL,
                                 stanc_args = character(),
                                 compile_args = character(),
                                 verbose = 1L,
                                 cache = TRUE) {
  if (!is.null(stan_file) && !is.null(code)) {
    stop("Provide exactly one of `stan_file` or `code`, not both.", call. = FALSE)
  }
  if (is.null(stan_file) && is.null(code)) {
    stop("Provide exactly one of `stan_file` or `code`.", call. = FALSE)
  }

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

  verbose <- as.integer(verbose)
  use_cache <- isTRUE(cache) &&
    !identical(Sys.getenv("NUTPIER_DISABLE_COMPILE_CACHE"), "1")

  bundle <- if (!is.null(code)) {
    inline_bundle(code)
  } else {
    file_bundle(normalizePath(stan_file, mustWork = TRUE))
  }

  if (use_cache) {
    compile_via_cache(bundle, stanc_args, compile_args, verbose)
  } else {
    compile_no_cache(bundle, stanc_args, compile_args, verbose)
  }
}

#' @export
print.nutpie_model <- function(x, ...) {
  cat("nutpie Stan model\n")
  src <- if (is.na(x$stan_file)) "<inline code>" else normalizePath(x$stan_file, mustWork = FALSE)
  cat("  Source: ", src, "\n")
  cat("  Library:", x$lib_path, "\n")
  invisible(x)
}
