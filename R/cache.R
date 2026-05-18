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

# Returns paths referenced by `#include` directives in stan_file (direct
# children only; use all_deps() for the transitive set). Handles bare,
# quoted, and angle-bracket forms with any amount of whitespace after
# `#include`. Stan `//` and `/* */` comments are stripped first so a
# commented-out directive doesn't get tracked as a real dependency, and
# the directive itself must be at the start of a (whitespace-only) line.
included_files <- function(stan_file) {
  text <- paste(readLines(stan_file, warn = FALSE), collapse = "\n")
  text <- gsub("(?s)/\\*.*?\\*/", "", text, perl = TRUE)
  text <- gsub("//[^\n]*", "", text, perl = TRUE)
  lines <- strsplit(text, "\n", fixed = TRUE)[[1L]]
  matches <- regmatches(lines, regexpr("^\\s*#include\\s+\\S+", lines))
  if (!length(matches)) return(character())
  raw <- sub("^\\s*#include\\s+", "", matches)
  file.path(dirname(stan_file), gsub('["<>]', "", raw))
}

canonical_path <- function(path) {
  normalizePath(path, mustWork = FALSE, winslash = "/")
}

# Walks #include directives transitively. Visited set is keyed on the
# canonical form so paths like `../foo.stan` and `foo.stan` resolved from
# sibling dirs don't double-count. Missing dependencies stay in the
# result -- the source bundle treats them as empty content so a deleted
# include still busts the hash.
all_deps <- function(stan_file, visited = character()) {
  key <- canonical_path(stan_file)
  if (key %in% visited) return(visited)
  visited <- c(visited, key)
  if (!file.exists(stan_file)) return(visited)
  for (inc in included_files(stan_file)) {
    visited <- all_deps(inc, visited)
  }
  visited
}

# Lowest common ancestor of a set of canonical paths, at path-component
# granularity. Used to figure out where to root the staged copy so each
# dep's relative position is preserved -- so `#include "../foo.stan"`
# still resolves under the cache dir.
common_root <- function(paths) {
  if (length(paths) == 0L) return(NULL)
  if (length(paths) == 1L) return(dirname(paths[1L]))
  parts <- strsplit(paths, "/", fixed = TRUE)
  min_len <- min(lengths(parts))
  if (min_len == 0L) return(NULL)
  shared <- character()
  for (i in seq_len(min_len)) {
    seg <- vapply(parts, function(p) p[[i]], character(1L))
    if (length(unique(seg)) != 1L) break
    shared <- c(shared, seg[[1L]])
  }
  if (length(shared) == 0L) return(NULL)
  root <- paste(shared, collapse = "/")
  if (!nzchar(root)) "/" else root
}

# Returns path with `root` stripped (root assumed to be a prefix). Both
# inputs must be canonical. Strips the trailing slash on root before
# comparing so root == "/foo" matches path "/foo/bar.stan".
relative_to <- function(path, root) {
  if (identical(root, "/")) {
    sub("^/", "", path)
  } else {
    prefix <- paste0(root, "/")
    if (!startsWith(path, prefix)) {
      stop("path ", path, " is not under root ", root, call. = FALSE)
    }
    substr(path, nchar(prefix) + 1L, nchar(path))
  }
}

# Returns the file's raw byte content, or NULL if the path doesn't
# exist. Raw bytes (not readLines() text) so the cache key is sensitive
# to trailing-newline differences, line-ending differences, and any
# encoding round-trip oddities -- a content cache should never falsely
# claim two byte-distinct files match. The NULL marker lets the bundle
# differentiate "present-and-empty" from "missing" in the hash (so
# deleting a `#include` busts the cache) while letting stage_bundle()
# skip writing the file so stanc surfaces a real "Could not find
# include file" error instead of silently treating the include as an
# empty no-op.
read_dep <- function(path) {
  if (file.exists(path)) {
    readBin(path, what = raw(), n = file.info(path)$size)
  } else {
    NULL
  }
}

# Build a source bundle = ordered list of (rel_path, content) pairs plus
# the rel_path of the main file to compile. For inline code there's a
# single synthetic file at `model.stan`; for file input we stage the
# transitive include set under the lowest common ancestor so relative
# `#include`s still resolve at compile time. `display_source` records
# what to show users (their original path, or NA for inline) since the
# main staged path under the cache dir isn't meaningful to them.
# `content` is always a raw vector (matching file_bundle), so the hash
# is computed over the same byte representation as the staged file.
inline_bundle <- function(code) {
  list(
    files = list(list(
      rel_path = INLINE_STAN,
      content  = charToRaw(enc2utf8(code))
    )),
    main = INLINE_STAN,
    display_source = NA_character_
  )
}

file_bundle <- function(stan_file) {
  deps <- all_deps(stan_file)
  root <- common_root(deps)
  if (is.null(root)) {
    stop(
      "Could not find a common root for `#include` dependencies of ",
      stan_file, ". Use absolute paths under a shared directory.",
      call. = FALSE
    )
  }
  files <- lapply(deps, function(p) {
    list(rel_path = relative_to(p, root), content = read_dep(p))
  })
  main_rel <- relative_to(canonical_path(stan_file), root)
  list(files = files, main = main_rel,
       display_source = canonical_path(stan_file))
}

# --- Hash key --------------------------------------------------------------

# Argument order is preserved (no sort): for override-style flags like
# `--name=foo --name=bar`, semantics are order-sensitive and we don't
# want the cache to silently coalesce them.
cache_key <- function(bundle, bs_version, stanc_args, compile_args) {
  # Sort files by rel_path so two equivalent bundles (e.g. different
  # walk orders of the same include tree) hash to the same key.
  # display_source is *not* hashed: two users compiling the same
  # source under different paths should share a cache slot.
  sorted <- bundle$files[order(vapply(bundle$files, `[[`, character(1L), "rel_path"))]
  payload <- list(
    files = sorted,
    main = bundle$main,
    bs_version = bs_version,
    stanc_args = as.character(stanc_args),
    compile_args = as.character(compile_args)
  )
  substr(digest::digest(payload, algo = "sha256"), 1L, 16L)
}

# Kept as a thin shim so existing internal tests against
# `inline_cache_key(code, ...)` (string-flavoured) still pass. The full
# bundle hash is accessed via `cache_key()`.
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
  ok    <- entry_ok_marker(entry)
  main  <- entry_main_path(entry, bundle$main)
  lib   <- entry_lib_path(entry, bundle$main)

  if (file.exists(ok) && file.exists(lib) && file.exists(main)) {
    if (verbose >= 1L) message("Using cached compiled model.")
    # Refresh the `ok` mtime so this entry counts as recently-used.
    # Pruning is LRU on the marker mtime; without the bump, a popular
    # but old-on-disk model could be auto-evicted right after a hit
    # returns it to the caller (issue surfaced in PR #24 review).
    Sys.setFileTime(ok, Sys.time())
    return(nutpie_model(
      lib_path      = normalizePath(lib,  mustWork = TRUE),
      stan_file     = bundle$display_source,
      staged_source = normalizePath(main, mustWork = TRUE)
    ))
  }

  # Wipe partial state from a prior failed compile so stage_bundle()
  # starts from a clean slate.
  if (dir.exists(entry)) {
    unlink(entry, recursive = TRUE, force = TRUE)
  }
  stage_bundle(bundle, entry)
  built <- compile_at_path(
    entry_main_path(entry, bundle$main),
    stanc_args, compile_args, verbose
  )
  file.create(ok)

  tryCatch(
    prune_cache_internal(CACHE_MAX_ENTRIES, CACHE_MIN_AGE_DAYS),
    error = function(e) NULL
  )

  nutpie_model(
    lib_path      = normalizePath(built, mustWork = TRUE),
    stan_file     = bundle$display_source,
    staged_source = normalizePath(
      entry_main_path(entry, bundle$main), mustWork = TRUE
    )
  )
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
#'   entries to retain. Defaults to 16.
#' @param min_age_days Minimum age (in days, by `ok` marker mtime) before
#'   an entry is eligible for eviction. Defaults to 14, so frequently
#'   re-used models aren't evicted just because the cache is hot.
#' @return Invisibly, the number of entries removed.
#' @examples
#' nutpie_prune_cache()
#' nutpie_prune_cache(max_entries = 8, min_age_days = 7)
#' @export
nutpie_prune_cache <- function(max_entries = 16L, min_age_days = 14L) {
  invisible(prune_cache_internal(
    as.integer(max_entries), as.numeric(min_age_days)
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
