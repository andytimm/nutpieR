test_that("interrupting a cold build stops make and its compiler before retry", {
  skip_on_os("windows") # This fixture delivers a POSIX SIGINT to R alone.
  skip_if(is.null(test_models$normal), "normal model fixture unavailable")
  dir <- withr::local_tempdir()
  make <- file.path(dir, "make")
  # Reuse the helper's real, pre-packaging model on success, so exercising
  # both publication and a cache hit costs no extra Stan/C++ compilation.
  normal <- test_models$normal
  lib <- paste0(tools::file_path_sans_ext(normal$staged_source), "_model.so")
  withr::local_envvar(c(
    PATH = paste(dir, Sys.getenv("PATH"), sep = .Platform$path.sep),
    R_USER_CACHE_DIR = dir, NUTPIER_DISABLE_COMPILE_CACHE = NA_character_,
    BUILD_LOG = dir, NORMAL_LIB = lib, R_PID = as.character(Sys.getpid())
  ))
  writeLines(c(
    "#!/bin/sh", 'set -eu',
    'echo "$1" >> "$BUILD_LOG/targets"',
    'case "$(cat "$BUILD_LOG/mode")" in',
    '  interrupt)',
    '    echo partial > "$1"',
    '    sleep 10 &', # safety bound if the caller ignores the interrupt
    '    child=$!',
    '    echo "$$ $child" > "$BUILD_LOG/pids"',
    '    kill -INT "$R_PID"', # only after both processes and the target exist
    '    wait "$child"',
    '    touch "$BUILD_LOG/finished" ;;',
    '  error) echo "deliberate compiler diagnostic" >&2; exit 2 ;;',
    '  success) cp "$NORMAL_LIB" "$1" ;;',
    'esac'
  ), make)
  Sys.chmod(make, "0755")
  compile <- function() nutpie_compile_model(stan_file = normal$stan_file, verbose = 0L)
  mode <- function(value) writeLines(value, file.path(dir, "mode"))
  targets <- function() readLines(file.path(dir, "targets"))

  mode("interrupt")
  interrupted <- tryCatch(compile(), interrupt = identity)
  expect_s3_class(interrupted, "interrupt")
  expect_false(file.exists(file.path(dir, "finished")))
  pids <- scan(file.path(dir, "pids"), integer(), quiet = TRUE)
  # Killing a process is asynchronous. Wait for exit, not a fixed delay;
  # a surviving compiler would run to the 10-second watchdog above.
  stopped <- function(pid) {
    tryCatch(ps::ps_wait(ps::ps_handle(pid), timeout = 1000L),
             no_such_process = function(e) TRUE)
  }
  expect_true(all(vapply(pids, stopped, logical(1))))
  entry <- dirname(dirname(targets()[1L]))
  expect_false(file.exists(nutpieR:::entry_ok_marker(entry)))
  expect_false(dir.exists(nutpieR:::entry_lock_path(entry)))

  mode("error")
  expect_error(compile(), "deliberate compiler diagnostic")
  expect_false(file.exists(nutpieR:::entry_ok_marker(entry)))
  mode("success")
  rebuilt <- compile()
  expect_s3_class(rebuilt, "nutpie_model")
  expect_true(file.exists(rebuilt$lib_path))
  expect_equal(compile()$lib_path, rebuilt$lib_path)
  # Interrupted build, genuine build error, successful retry; no fourth make.
  expect_equal(targets(), rep(targets()[1L], 3L))
})
