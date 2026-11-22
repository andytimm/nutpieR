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
#' For either `stan_file = ...` or inline `code = ...`, nutpieR asks the
#' bundled `stanc` compiler to resolve transitive `#include` files using the
#' same ordered include paths as compilation. Their bytes are part of the
#' cache key, so editing an included file (or the main file itself) triggers a
#' recompile. Include-bearing models are staged as stanc-expanded source. This
#' makes nested and external includes reliable, but compiler errors for those
#' models refer to the staged expanded source rather than an original include
#' line. Unusual stanc output-mode or make/compiler overrides that cannot be
#' tracked are compiled fresh without using the persistent cache. For a
#' file-based model, that conservative route compiles in the source directory
#' under a per-source lock and returns a copied, unique temporary library path.
#' It requires that source directory to be writable and contain no spaces;
#' otherwise nutpieR stops with guidance rather than using different compiler
#' semantics.
#'
#' The cache is bounded by [`nutpie_prune_cache()`][nutpie_prune_cache],
#' which runs automatically after a new persistent cache entry is built
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
#' left over from earlier versions can be deleted. The normal cached/staged
#' route no longer reads or writes them; the conservative untracked file route
#' can still leave a source-adjacent model library and `.hpp` file.
#'
#' @param stan_file Path to a `.stan` file. Exactly one of `stan_file` or
#'   `code` must be provided.
#' @param code A string containing Stan model code.
#' @param stanc_args Character vector of extra arguments passed to the
#'   `stanc` compiler (e.g., `"--O1"` for optimization). Repeated
#'   `--include-paths=` arguments keep their supplied order. Do not use stanc
#'   output-mode arguments such as `--auto-format` here; if supplied with an
#'   include model, nutpieR compiles fresh rather than risking a cache hit.
#' @param compile_args Character vector of extra arguments passed to `make`
#'   during compilation. On macOS, nutpieR keeps Stan's fast process-wide
#'   `tbbmalloc_proxy` allocator but patches its bundled source to be safe
#'   (GitHub #36); this is automatic and idempotent. Set
#'   `NUTPIER_NO_TBB_PROXY_PATCH=1` to skip the patch, or pass
#'   `"TBB_LIBRARIES=tbb"` here to drop the proxy entirely (safe, but ~17%
#'   slower on large, many-chain models). If an unpatched proxy is loaded,
#'   nutpieR stops sampling and asks you to restart R because all subsequent R
#'   allocations in that process are unsafe.
#' @param verbose Integer controlling compilation output. `0` = silent,
#'   `1` (default) = print status messages.
#'   Compiler output is captured and shown if the build fails.
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

  verbose <- check_count(verbose, "verbose", min = 0L)
  cache <- check_flag(cache, "cache")
  use_cache <- cache &&
    !identical(Sys.getenv("NUTPIER_DISABLE_COMPILE_CACHE"), "1")

  source_path <- if (is.null(stan_file)) NULL else normalizePath(stan_file, mustWork = TRUE)
  stanc_args <- normalize_stanc_include_paths(stanc_args)
  bundle <- bundle_for_compile(source_path, code, stanc_args, compile_args)

  # Compiler/make overrides or stanc resolution failures can prevent reliable
  # dependency tracking and source expansion. Never claim a cache hit then:
  # a fresh build is slower but cannot silently reuse an old model.
  if (isTRUE(attr(bundle, "untracked_includes"))) {
    warning(
      "Could not safely track #include dependencies with the current compiler/make settings; ",
      "compiling without the persistent cache.",
      call. = FALSE
    )
    return(compile_no_cache_untracked(
      bundle, source_path, stanc_args, compile_args, verbose
    ))
  }

  if (use_cache) {
    compile_via_cache(bundle, stanc_args, compile_args, verbose)
  } else {
    compile_no_cache(bundle, stanc_args, compile_args, verbose)
  }
}

# Keep make's target, user arguments, and single STANCFLAGS assignment in
# BridgeStan order. Do not shell-quote argv: processx handles Windows quoting.
compile_stan_model <- function(stan_file, stanc_args, compile_args) {
  if (!endsWith(stan_file, ".stan")) stop("File must be a .stan file", call. = FALSE)
  target <- paste0(sub("[.]stan$", "", stan_file), "_model.so")
  flags <- c(paste0("--include-paths=", dirname(stan_file)), stanc_args)
  args <- c(target, compile_args, paste0("STANCFLAGS=", paste(flags, collapse = " ")))
  wd <- prepare_bridgestan_build()
  logs <- tempfile(c("nutpieR-make-out-", "nutpieR-make-err-"))
  on.exit(unlink(logs), add = TRUE)
  suspendInterrupts({
    make <- processx::process$new(
      "make", args, wd = wd, env = c("current", STAN_THREADS = "true"),
      stdout = logs[1L], stderr = logs[2L], cleanup_tree = TRUE, windows_hide_window = TRUE
    )
    on.exit(suspendInterrupts({
      # kill() stops the Unix process group, including macOS system tools
      # whose environment marker is hidden from kill_tree().
      make$kill()
      make$kill_tree()
    }), add = TRUE, after = FALSE)
  })
  # Service the R console's event loop even when make produces no output.
  while (make$is_alive()) Sys.sleep(0.05)
  status <- make$get_exit_status()
  if (status != 0L) {
    stop("Model compilation failed (make exit ", status, "):\n",
         paste(c(readLines(logs[2L], warn = FALSE), readLines(logs[1L], warn = FALSE)),
               collapse = "\n"), call. = FALSE)
  }
  private_tbb_model(target)
}

#' @export
print.nutpie_model <- function(x, ...) {
  cat("nutpie Stan model\n")
  src <- if (is.na(x$stan_file)) "<inline code>" else normalizePath(x$stan_file, mustWork = FALSE)
  cat("  Source: ", src, "\n")
  cat("  Library:", x$lib_path, "\n")
  invisible(x)
}
