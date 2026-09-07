# Content-hashed compile cache for nutpie_compile_model().
#
# Why a hashed cache instead of compiling in-place next to the .stan?
# Once a shared library has been loaded at a given path in the current
# process, dlopen(3) returns the cached library on every subsequent load
# of that path -- even if the on-disk file has been overwritten. So a
# user who recompiles after editing their .stan would silently keep
# sampling from the old logp. (See GitHub issue #23, and bridgestan
# issue #36 for the upstream-side acknowledgment.)
#
# By keying the artifact directory on a content hash, every edited source
# lands at a fresh path -- dlopen has never seen it -- so the new library
# is loaded for real. Unchanged sources hit the cache.
#
# This mirrors Python nutpie's design: each compiled model lives under
#   tools::R_user_dir("nutpieR", "cache")/models/<hash16>/
# with the source(s) staged under src/ and a post-compile `ok` marker so
# an interrupted compile is never reused. Eviction policy also mirrors
# nutpie: 16-entry cap, 14-day minimum age, oldest-eligible-first.

INLINE_STAN <- "model.stan"
CACHE_MAX_ENTRIES <- 16L
CACHE_MIN_AGE_DAYS <- 14L

.cache_state <- new.env(parent = emptyenv())

# --- Source bundle construction --------------------------------------------

# Cache format version.  Bump this whenever the interpretation of a source
# bundle changes: otherwise a library built from the old staging scheme could
# be mistaken for one built from the new scheme.
CACHE_SCHEMA_VERSION <- 2L

canonical_path <- function(path) {
  normalizePath(path, mustWork = FALSE, winslash = "/")
}

read_dep <- function(path) {
  if (file.exists(path)) {
    readBin(path, what = raw(), n = file.info(path)$size)
  } else {
    NULL
  }
}

# Do not parse Stan includes in R.  stanc owns that grammar and, importantly,
# its search rule is based on the main source root rather than the directory of
# the file containing a nested include.  This cheap gate merely preserves the
# ordinary no-include fast path; a false positive only runs stanc's resolver.
has_possible_include <- function(content) {
  # stanc accepts directives after other tokens on a line (for example,
  # `functions { #include f.stan }`).  This deliberately broad token search
  # has false positives in comments but no known syntax-shaped false negatives.
  grepl("#include", rawToChar(content), fixed = TRUE)
}

source_bundle <- function(content, main = INLINE_STAN, display_source = NA_character_) {
  list(
    files = list(list(rel_path = main, content = content)),
    main = main,
    display_source = display_source,
    dependencies = list()
  )
}

inline_bundle <- function(code) {
  source_bundle(charToRaw(enc2utf8(code)))
}

file_bundle <- function(stan_file) {
  stan_file <- canonical_path(stan_file)
  source_bundle(
    read_dep(stan_file),
    main = basename(stan_file),
    display_source = stan_file
  )
}

stanc_system <- function(stanc, args) {
  # system2 invokes a shell on some R platforms.  Quote every complete
  # argument, not just paths, so an include directory with spaces remains one
  # --include-paths value.  Arguments are still passed in their supplied order.
  # Keep stderr out of stdout: --info and --auto-format stdout is structured
  # JSON/source and compiler warnings must never become cached Stan text.
  err_file <- tempfile("nutpieR-stanc-stderr-")
  on.exit(unlink(err_file, force = TRUE), add = TRUE)
  output <- suppressWarnings(system2(
    stanc,
    args = vapply(args, shQuote, character(1L)),
    stdout = TRUE,
    stderr = err_file
  ))
  diagnostics <- if (file.exists(err_file)) readLines(err_file, warn = FALSE) else character()
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop(paste(c(output, diagnostics), collapse = "\n"), call. = FALSE)
  }
  output
}

# Run a short synchronous stanc subprocess in BridgeStan's working directory
# without making the test-only `withr` package a runtime dependency.
with_stanc_workdir <- function(path, expr) {
  old <- getwd()
  on.exit(setwd(old), add = TRUE)
  setwd(path)
  force(expr)
}

# BridgeStan puts include flags into a make shell command. On Windows,
# backslashes in path values are separators, not shell escapes. Normalize only
# these flags; other arguments must retain the conservative tracking checks.
normalize_stanc_include_paths <- function(stanc_args,
                                          windows = .Platform$OS.type == "windows") {
  args <- as.character(stanc_args)
  if (windows) {
    include <- startsWith(args, "--include-paths=")
    args[include] <- gsub("\\", "/", args[include], fixed = TRUE)
  }
  args
}

# BridgeStan's compile_model prepends --include-paths=<main source directory>
# before the supplied flags.  Repeat that exact ordered prefix for stanc's
# resolver.  stanc accumulates repeated include-paths flags, so do not combine,
# sort, or de-duplicate user flags.
stanc_args_for_source <- function(stan_file, stanc_args) {
  c(
    paste0("--include-paths=", dirname(canonical_path(stan_file))),
    as.character(stanc_args)
  )
}

# A few output-mode switches are not normal compile overrides.  Combining one
# with --info/--auto-format changes stanc's output contract, so do not infer a
# cache key from it.  The caller takes the conservative uncached path instead.
stanc_make_override_present <- function(stanc = NULL) {
  # MAKEFLAGS can inject variable assignments or select another makefile, and
  # MAKEFILES loads make fragments before BridgeStan's Makefile.  Either can
  # select a compiler with include rules different from bundled stanc.
  if (nzchar(Sys.getenv("MAKEFLAGS", unset = "")) ||
      nzchar(Sys.getenv("MAKEFILES", unset = "")) ||
      nzchar(Sys.getenv("STANC", unset = "")) ||
      nzchar(Sys.getenv("STANCFLAGS", unset = ""))) {
    return(TRUE)
  }
  if (is.null(stanc)) {
    stanc <- tryCatch(bridgestan_stanc_path(), error = function(e) NULL)
  }
  if (is.null(stanc)) return(TRUE)
  # A non-empty user make/local can change stanc or its inputs by arbitrary
  # make logic.  It is deliberately treated as untrackable rather than trying
  # to parse a makefile in R.
  local <- file.path(dirname(dirname(stanc)), "make", "local")
  file.exists(local) && isTRUE(file.info(local)$size > 0L)
}
stanc_tracking_supported <- function(stanc_args, compile_args = character()) {
  args <- as.character(stanc_args)
  output_mode <- any(grepl(
    "^(--(?:auto-format|print-canonical|print-cpp|info)(?:=|$)|--output(?:=|$)|-o(?:$|.))",
    args,
    perl = TRUE
  ))
  # BridgeStan joins STANCFLAGS into make's shell command.  Quoted or
  # whitespace-containing elements can therefore mean something different
  # from an argv-based system2 invocation; use the direct uncached compiler
  # route instead of deriving a possibly different dependency set.
  # Only accept arguments whose characters have identical plain-word meaning
  # in BridgeStan's make-shell STANCFLAGS string and in system2 argv.  Shell
  # expansion/quoting characters ($, backticks, backslashes, globs, pipes,
  # redirects, etc.) take the direct uncached route.
  # An embedded tilde is literal (notably RUNNER~1 in Windows 8.3 paths).
  # Keep leading/path-list tildes conservative: they can request shell expansion.
  shell_sensitive <- any(!grepl("^[[:alnum:]_./,:=+@%~-]+$", args)) ||
    any(grepl("(^|[=:,])~", args))
  make_overrides <- any(grepl(
    paste0(
      "^(?:STANC|STANCFLAGS|MAKEFILES|MAKEFLAGS)(?::|\\+|\\?)?=",
      "|^-f|^--(?:file|makefile)(?:=|$)|^--eval(?:=|$)|^-E(?:.|$)"
    ),
    as.character(compile_args), perl = TRUE
  )) || stanc_make_override_present()
  !(output_mode || shell_sensitive || make_overrides)
}

# Ask the bundled stanc itself for both the resolved dependency list and the
# canonical source with #includes expanded.  The expansion makes the staged
# compilation independent of the cache directory's layout, including nested
# includes found through the main source root or ordered external search paths.
# It intentionally means compiler diagnostics for include models refer to the
# staged expanded source rather than an original include line; stanc exposes no
# source-map format for preserving those locations.
resolve_included_source <- function(stan_file, stanc_args,
                                    compile_args = character()) {
  stanc_args <- normalize_stanc_include_paths(stanc_args)
  if (!stanc_tracking_supported(stanc_args, compile_args)) {
    return(NULL)
  }
  stanc <- bridgestan_stanc_path()
  common <- stanc_args_for_source(stan_file, stanc_args)
  # BridgeStan runs make (and therefore stanc) from its downloaded source
  # directory.  Use the same cwd as well as the same argument order.
  stanc_run <- function(args) {
    with_stanc_workdir(dirname(dirname(stanc)), stanc_system(stanc, args))
  }
  info <- tryCatch(
    stanc_run(c("--info", "--color=never", common, stan_file)),
    error = identity
  )
  if (inherits(info, "error")) {
    # A missing include is a real compiler error, not an excuse to reuse an
    # old entry.  Returning the error lets callers see stanc's own diagnostic.
    stop(conditionMessage(info), call. = FALSE)
  }
  parsed <- tryCatch(
    jsonlite::fromJSON(paste(info, collapse = "\n"), simplifyVector = FALSE),
    error = identity
  )
  if (inherits(parsed, "error") || is.null(parsed$included_files)) {
    return(NULL)
  }
  expanded <- tryCatch(
    stanc_run(c(
      "--auto-format", "--canonicalize=includes", "--color=never",
      common, stan_file
    )),
    error = identity
  )
  if (inherits(expanded, "error")) return(NULL)

  included <- vapply(parsed$included_files, canonical_path, character(1L))
  # Keep the bytes separately from the expanded source.  This deliberately
  # invalidates on every dependency-byte edit, even edits stripped by the
  # formatter, and records a deleted dependency as NULL rather than a cache hit.
  dependencies <- lapply(sort(unique(included)), function(path) {
    list(path = path, content = read_dep(path))
  })
  list(
    content = charToRaw(enc2utf8(paste(expanded, collapse = "\n"))),
    dependencies = dependencies
  )
}

# Build a bundle for the requested source.  The normal path does no subprocess
# work.  Include-bearing sources are resolved and expanded by stanc before they
# are staged, exactly avoiding a second, hand-written approximation of stanc's
# include rules.
bundle_for_compile <- function(stan_file = NULL, code = NULL, stanc_args = character(),
                               compile_args = character()) {
  raw <- if (is.null(stan_file)) inline_bundle(code) else file_bundle(stan_file)
  if (!has_possible_include(raw$files[[1L]]$content)) return(raw)

  analysis_dir <- NULL
  analysis_file <- stan_file
  if (is.null(analysis_file)) {
    analysis_dir <- tempfile("nutpieR-include-analysis-")
    dir.create(analysis_dir, recursive = TRUE)
    analysis_file <- file.path(analysis_dir, INLINE_STAN)
    writeBin(raw$files[[1L]]$content, analysis_file)
    on.exit(unlink(analysis_dir, recursive = TRUE, force = TRUE), add = TRUE)
  }
  resolved <- resolve_included_source(analysis_file, stanc_args, compile_args)
  if (is.null(resolved)) {
    attr(raw, "untracked_includes") <- TRUE
    return(raw)
  }
  # Retain the original main bytes as well.  The expanded source is what is
  # compiled, while this keeps the cache's byte-sensitive invalidation promise
  # for comments/formatting in the main file and avoids an inline temporary
  # analysis path becoming part of the key.
  main_dependency <- list(
    path = if (is.null(stan_file)) INLINE_STAN else canonical_path(stan_file),
    content = raw$files[[1L]]$content
  )
  raw$files[[1L]]$content <- resolved$content
  raw$dependencies <- c(list(main_dependency), resolved$dependencies)
  raw
}

# --- Hash key --------------------------------------------------------------

cache_key <- function(bundle, bs_version, stanc_args, compile_args) {
  sorted <- bundle$files[order(vapply(bundle$files, `[[`, character(1L), "rel_path"))]
  payload <- list(
    schema = CACHE_SCHEMA_VERSION,
    files = sorted,
    dependencies = bundle$dependencies,
    main = bundle$main,
    bs_version = bs_version,
    stanc_args = as.character(stanc_args),
    compile_args = as.character(compile_args)
  )
  substr(digest::digest(payload, algo = "sha256"), 1L, 16L)
}

inline_cache_key <- function(content, bs_version, stanc_args, compile_args) {
  cache_key(inline_bundle(content), bs_version, stanc_args, compile_args)
}

# --- Cache layout ----------------------------------------------------------

dir_writable <- function(path) {
  dir.exists(path) && file.access(path, mode = 2L) == 0L
}

bs_version <- function() {
  if (is.null(.cache_state$bs_version)) {
    .cache_state$bs_version <- bridgestan_version()
  }
  .cache_state$bs_version
}

# Cache root, memoized per-process (re-resolved if R_USER_CACHE_DIR flips
# between calls). Falls back to a tempdir if R_user_dir is unwritable;
# the warning fires once per session.
cache_root <- function() {
  env <- Sys.getenv("R_USER_CACHE_DIR", unset = "")
  if (!is.null(.cache_state$root) &&
      identical(.cache_state$root_env, env)) {
    return(.cache_state$root)
  }

  base <- tools::R_user_dir("nutpieR", "cache")
  suppressWarnings(dir.create(base, recursive = TRUE))
  if (!dir_writable(base)) {
    if (!isTRUE(getOption("nutpieR.warned_cache_fallback"))) {
      warning(
        "Could not write to R_user_dir for nutpieR cache; falling back ",
        "to a session-scoped tempdir. Cached models will not persist ",
        "across R sessions.", call. = FALSE
      )
      options(nutpieR.warned_cache_fallback = TRUE)
    }
    base <- file.path(tempdir(), "nutpieR-cache")
    dir.create(base, showWarnings = FALSE, recursive = TRUE)
  }
  models <- file.path(base, "models")
  dir.create(models, showWarnings = FALSE, recursive = TRUE)
  .cache_state$root <- models
  .cache_state$root_env <- env
  models
}

# Public alias retained for back-compat with code that calls it directly.
inline_cache_dir <- function() cache_root()

# Paths within a cache entry directory.
entry_src_dir   <- function(entry) file.path(entry, "src")
entry_ok_marker <- function(entry) file.path(entry, "ok")
entry_main_path <- function(entry, main_rel) {
  file.path(entry_src_dir(entry), main_rel)
}
entry_lib_path  <- function(entry, main_rel) {
  paste0(tools::file_path_sans_ext(entry_main_path(entry, main_rel)),
         "_model.so")
}

# A cache key may be requested by several R processes at once (e.g. parallel
# CI jobs sharing R_USER_CACHE_DIR). `dir.create()` is atomic on supported
# local filesystems, so a sibling lock directory serialises staging and
# compiling of one entry without serialising unrelated models. A stale lock
# from a killed process is reclaimed after half a day; an active compiler
# should finish well before then, while a waiter times out with an actionable
# error rather than blocking indefinitely.
CACHE_LOCK_TIMEOUT_SECS <- 600
CACHE_LOCK_STALE_SECS <- 12 * 60 * 60

entry_lock_path <- function(entry) paste0(entry, ".lock")

with_cache_entry_lock <- function(entry, expr,
                                  timeout_secs = CACHE_LOCK_TIMEOUT_SECS,
                                  stale_secs = CACHE_LOCK_STALE_SECS) {
  lock <- entry_lock_path(entry)
  started <- Sys.time()
  repeat {
    if (dir.create(lock, showWarnings = FALSE)) break

    info <- file.info(lock)
    age <- as.numeric(difftime(Sys.time(), info$mtime, units = "secs"))
    if (dir.exists(lock) && is.finite(age) && age > stale_secs) {
      unlink(lock, recursive = TRUE, force = TRUE)
      next
    }
    waited <- as.numeric(difftime(Sys.time(), started, units = "secs"))
    if (!is.finite(waited) || waited >= timeout_secs) {
      stop(
        "Timed out waiting for another process to compile the same cached ",
        "Stan model. If no compilation is running, remove stale lock: ", lock,
        call. = FALSE
      )
    }
    Sys.sleep(0.05)
  }
  on.exit(unlink(lock, recursive = TRUE, force = TRUE), add = TRUE)
  force(expr)
}

cached_model <- function(entry, main_rel, display_source, verbose) {
  ok <- entry_ok_marker(entry)
  main <- entry_main_path(entry, main_rel)
  lib <- entry_lib_path(entry, main_rel)
  if (!file.exists(ok) || !file.exists(lib) || !file.exists(main)) {
    return(NULL)
  }
  if (verbose >= 1L) message("Using cached compiled model.")
  # A cache hit never calls into Rust, so ensure_safe_tbb_proxy (which only
  # runs during a real compile) would never re-patch a stale, unpatched
  # tbbmalloc_proxy for an upgrading user whose model is already cached.
  ensure_tbb_proxy_patched()
  # Marker mtime is the LRU timestamp for pruning.
  Sys.setFileTime(ok, Sys.time())
  nutpie_model(
    lib_path = normalizePath(lib, mustWork = TRUE),
    stan_file = display_source,
    staged_source = normalizePath(main, mustWork = TRUE)
  )
}

# --- Compile --------------------------------------------------------------

nutpie_model <- function(lib_path, stan_file, staged_source) {
  structure(
    list(
      lib_path = lib_path,
      stan_file = stan_file,
      staged_source = staged_source
    ),
    class = "nutpie_model"
  )
}

# Materialise a bundle into dest_dir/src/. Each file lands at the
# rel_path declared in the bundle so relative `#include`s resolve.
# Files with NULL content (missing deps) are deliberately not written
# so stanc fails naturally with its own "include not found" message.
# writeBin (not writeLines) so bytes hashed == bytes staged, with no
# platform line-ending translation or trailing-newline insertion.
stage_bundle <- function(bundle, dest_dir) {
  src_root <- entry_src_dir(dest_dir)
  for (f in bundle$files) {
    if (is.null(f$content)) next
    target <- file.path(src_root, f$rel_path)
    dir.create(dirname(target), showWarnings = FALSE, recursive = TRUE)
    writeBin(f$content, target)
  }
}

# Compile a staged stan_file via bridgestan. Wraps the platform-specific
# path massaging needed on Windows (spaces, backslashes) and surfaces
# timing for verbose >= 1. Returns the absolute path of the produced
# `_model.so`.
#
# Removes any existing .hpp first so a stanc_args change actually
# re-runs stanc; otherwise make sees `.hpp` is up-to-date and skips
# regeneration, producing a stale binary.
compile_at_path <- function(stan_file, stanc_args, compile_args, verbose) {
  base <- tools::file_path_sans_ext(stan_file)
  unlink(paste0(base, ".hpp"), force = TRUE)

  build_path <- stan_file
  if (.Platform$OS.type == "windows") {
    # On Windows, make + stanc both prefer forward slashes -- backslashes
    # get eaten by the shell when bridgestan invokes make with STANCFLAGS,
    # which then breaks `--include-paths=...`. shortPathName is only
    # applied if the dir has spaces (make can't handle those); it must
    # target the *directory* only (it would truncate ".stan" to ".STA"
    # on the basename, which bridgestan rejects).
    if (grepl(" ", dirname(build_path))) {
      short_dir <- utils::shortPathName(dirname(build_path))
      build_path <- file.path(short_dir, basename(build_path))
    }
    build_path <- gsub("\\\\", "/", build_path)
  }
  if (grepl(" ", build_path)) {
    stop(
      "Could not resolve a no-space build path for ", stan_file, ". ",
      "On Windows, ensure 8.3 short names are enabled on the volume, or ",
      "set R_USER_CACHE_DIR to a no-space path.",
      call. = FALSE
    )
  }

  if (verbose >= 1L) {
    message("Compiling Stan model...")
    start_time <- proc.time()[["elapsed"]]
  }
  built_so <- compile_stan_model(
    build_path,
    as.character(stanc_args),
    as.character(compile_args)
  )
  if (verbose >= 1L) {
    message(sprintf("Compiled in %.1fs",
                    proc.time()[["elapsed"]] - start_time))
  }
  normalizePath(built_so, mustWork = TRUE)
}

# Cache-aware compile. On hit returns the cached artifact; on miss
# stages, compiles, writes the `ok` marker, and opportunistically prunes
# the cache root down to CACHE_MAX_ENTRIES.
#
# A cache entry counts as "ready" only if BOTH the `_model.so` and the
# `ok` marker are present. The marker is written *after* a successful
# compile, so a Ctrl-C'd previous session leaves the entry incomplete
# and the next call recompiles.
compile_via_cache <- function(bundle, stanc_args, compile_args, verbose) {
  key <- cache_key(bundle, bs_version(), stanc_args, compile_args)
  entry <- file.path(cache_root(), key)

  # Fast path: a complete entry is immutable, so it is safe to use without
  # taking the per-key compile lock.
  hit <- cached_model(entry, bundle$main, bundle$display_source, verbose)
  if (!is.null(hit)) return(hit)

  # A waiter must check again after it owns the lock: the prior lock holder may
  # have completed the same build while this process waited.
  with_cache_entry_lock(entry, {
    hit <- cached_model(entry, bundle$main, bundle$display_source, verbose)
    if (!is.null(hit)) {
      hit
    } else {
      # Wipe partial state from a prior failed compile so stage_bundle() starts
      # from a clean slate. The lock prevents another process from observing or
      # changing this entry until the `ok` marker is written.
      if (dir.exists(entry)) {
        unlink(entry, recursive = TRUE, force = TRUE)
      }
      stage_bundle(bundle, entry)
      built <- compile_at_path(
        entry_main_path(entry, bundle$main),
        stanc_args, compile_args, verbose
      )
      file.create(entry_ok_marker(entry))

      tryCatch(
        prune_cache_internal(CACHE_MAX_ENTRIES, CACHE_MIN_AGE_DAYS),
        error = function(e) NULL
      )

      nutpie_model(
        lib_path = normalizePath(built, mustWork = TRUE),
        stan_file = bundle$display_source,
        staged_source = normalizePath(
          entry_main_path(entry, bundle$main), mustWork = TRUE
        )
      )
    }
  })
}

# cache = FALSE escape hatch: stage + compile in a fresh tempdir,
# leaving the persistent cache alone. Used when the caller wants to
# force a fresh compile without polluting (or evicting from) the cache.
compile_no_cache <- function(bundle, stanc_args, compile_args, verbose) {
  dest <- tempfile("nutpieR-build-")
  dir.create(dest, recursive = TRUE)
  stage_bundle(bundle, dest)
  built <- compile_at_path(
    entry_main_path(dest, bundle$main),
    stanc_args, compile_args, verbose
  )
  nutpie_model(
    lib_path      = normalizePath(built, mustWork = TRUE),
    stan_file     = bundle$display_source,
    staged_source = normalizePath(
      entry_main_path(dest, bundle$main), mustWork = TRUE
    )
  )
}

# Compile an include-bearing source only when compiler tracking is unavailable.
# The fresh directory guarantees a new library path.  For file input, make the
# original main-source root the next include root after BridgeStan's empty
# staging root, preserving its normal parent-before-user search precedence as
# closely as possible without pretending that an unsafe cache key is valid.
compile_no_cache_untracked <- function(bundle, source_path, stanc_args,
                                       compile_args, verbose) {
  if (is.null(source_path)) {
    return(compile_no_cache(bundle, stanc_args, compile_args, verbose))
  }
  source_dir <- dirname(source_path)
  if (grepl(" ", source_path, fixed = TRUE) || file.access(source_dir, 2L) != 0L) {
    stop(
      "Cannot safely use an untrackable compiler/make override for this ",
      "include model: its source directory must be writable and have no spaces. ",
      "Remove the override or use a source path meeting these requirements.",
      call. = FALSE
    )
  }

  # A custom stanc/make override can have include semantics that the bundled
  # resolver cannot prove.  Compile the original file in its original parent
  # directory so BridgeStan prepends exactly that source root.  Serialise this
  # source-adjacent build across R processes: both stanc's .hpp and BridgeStan's
  # .so name are fixed by the source basename.  Copy only the result to a fresh
  # tempdir before returning it, so callers never receive a reusable dlopen path.
  # This rare conservative route may leave BridgeStan's normal .hpp/.so build
  # by-products beside the source, just like a direct BridgeStan compilation.
  lock_entry <- file.path(
    source_dir,
    paste0(".nutpieR-untracked-", digest::digest(canonical_path(source_path), algo = "sha256"))
  )
  with_cache_entry_lock(lock_entry, {
    built <- compile_at_path(source_path, stanc_args, compile_args, verbose)
    dest <- tempfile("nutpieR-untracked-build-")
    dir.create(dest, recursive = TRUE)
    copied <- file.path(dest, basename(built))
    if (!file.copy(built, copied, overwrite = TRUE)) {
      stop("Could not copy fresh untracked Stan library to ", copied, call. = FALSE)
    }
    nutpie_model(
      lib_path = normalizePath(copied, mustWork = TRUE),
      stan_file = bundle$display_source,
      staged_source = source_path
    )
  })
}
# --- Pruning --------------------------------------------------------------

# Mirrors nutpie's policy: cap the cache at `max_entries` valid entries,
# but only evict entries at least `min_age_days` old (by `ok` marker
# mtime). Among eligible entries, oldest first. Returns the number of
# entries removed.
prune_cache_internal <- function(max_entries, min_age_days) {
  root <- cache_root()
  entries <- list.dirs(root, recursive = FALSE)
  if (!length(entries)) return(0L)

  ok_paths <- vapply(entries, entry_ok_marker, character(1L))
  has_ok   <- file.exists(ok_paths)
  valid_entries  <- entries[has_ok]
  valid_ok_paths <- ok_paths[has_ok]
  if (length(valid_entries) <= max_entries) return(0L)

  mtimes <- file.info(valid_ok_paths)$mtime
  age_days <- as.numeric(difftime(Sys.time(), mtimes, units = "days"))
  eligible_idx <- which(age_days >= min_age_days)
  if (!length(eligible_idx)) return(0L)

  over_cap <- length(valid_entries) - max_entries
  ord <- eligible_idx[order(mtimes[eligible_idx])]
  to_remove <- ord[seq_len(min(over_cap, length(ord)))]
  for (i in to_remove) {
    unlink(valid_entries[i], recursive = TRUE, force = TRUE)
  }
  length(to_remove)
}

#' Prune the nutpieR compile cache
#'
#' Evicts older entries from the nutpieR compile cache so it stays
#' bounded. Mirrors Python nutpie's policy: cap the cache at
#' `max_entries`, but only evict entries at least `min_age_days` old
#' (oldest first). Called automatically at the end of every successful
#' compile -- you usually don't need to invoke it directly, but it's
#' here for one-off manual cleanup or scripted maintenance.
#'
#' @param max_entries Maximum number of valid (fully compiled) cache
#'   entries to retain. Must be a non-negative whole number. Defaults to 16.
#' @param min_age_days Minimum age (in days, by `ok` marker mtime) before
#'   an entry is eligible for eviction. Must be a non-negative finite number.
#'   Defaults to 14, so frequently re-used models aren't evicted just because
#'   the cache is hot.
#' @return Invisibly, the number of entries removed.
#' @examples
#' nutpie_prune_cache()
#' nutpie_prune_cache(max_entries = 8, min_age_days = 7)
#' @export
nutpie_prune_cache <- function(max_entries = 16L, min_age_days = 14L) {
  max_entries <- check_count(max_entries, "max_entries", min = 0L)
  if (length(min_age_days) != 1L) {
    stop("`min_age_days` must be a single non-negative finite number.",
         call. = FALSE)
  }
  if (!is.numeric(min_age_days) || !is.finite(min_age_days) ||
      min_age_days < 0) {
    stop("`min_age_days` must be a non-negative finite number.",
         call. = FALSE)
  }
  invisible(prune_cache_internal(
    max_entries, as.numeric(min_age_days)
  ))
}

#' Clear the nutpieR compile cache
#'
#' Removes the current resolved compile cache tree under
#' [`nutpie_cache_dir()`][nutpie_cache_dir]. Cached compiled models will
#' be recompiled on next use.
#'
#' @section Warning:
#'
#' This deletes the underlying `_model.so` files. If you hold a
#' `nutpie_model` object whose library hasn't been opened yet (no prior
#' [`nutpie_sample()`][nutpie_sample] call on it), its `lib_path` will
#' point at a deleted file and subsequent use will fail. Models that
#' were already opened in the current session keep working — once
#' loaded, the OS retains the mapped library independently of the file
#' on disk.
#'
#' Only the *active* cache root is cleared. If `R_USER_CACHE_DIR` was
#' previously unset (or pointed somewhere else) and a different root
#' was resolved earlier in the session, that older directory is left
#' alone so models still backed by it remain valid.
#'
#' @return Invisibly `NULL`.
#' @examples
#' nutpie_clear_cache()
#' @export
nutpie_clear_cache <- function() {
  root <- cache_root()
  if (dir.exists(root)) unlink(root, recursive = TRUE, force = TRUE)
  rm(list = ls(.cache_state), envir = .cache_state)
  invisible(NULL)
}

#' Path to the nutpieR compile cache directory
#'
#' Returns the directory under which `nutpie_compile_model()` stores its
#' content-hashed compile artifacts (one subdirectory per unique
#' source + flags + BridgeStan version). Useful for inspection,
#' troubleshooting, or `unlink()`-ing a single entry.
#'
#' @return A character string with the path to the cache root.
#' @examples
#' nutpie_cache_dir()
#' @export
nutpie_cache_dir <- function() {
  cache_root()
}
