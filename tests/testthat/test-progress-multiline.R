multiline_snapshot <- function(draws = c(80L, 40L, 0L),
                               gradients = draws * 100,
                               tuning = c(TRUE, TRUE, TRUE),
                               started = c(TRUE, TRUE, FALSE),
                               divergences = c(0L, 0L, 0L)) {
  lapply(seq_along(draws), function(chain) list(
    chain = chain, finished_draws = draws[[chain]], total_draws = 200L,
    divergences = divergences[[chain]], tuning = tuning[[chain]],
    started = started[[chain]], latest_num_steps = 7L,
    total_num_steps = draws[[chain]] * 7,
    total_num_gradients = gradients[[chain]], step_size = 0.125,
    runtime = 1, divergent_draws = integer()
  ))
}

multiline_recorder <- function() {
  testthat::local_mocked_bindings(
    progress_messages_muffled = function() FALSE,
    .package = "nutpieR", .env = parent.frame()
  )
  state <- new.env(parent = emptyenv())
  state$updates <- list()
  state$done <- list()
  state$update <- function(...) {
    state$updates[[length(state$updates) + 1L]] <- list(...)
  }
  state$finish <- function(...) {
    state$done[[length(state$done) + 1L]] <- list(...)
  }
  state$latest <- function(id) {
    calls <- Filter(function(x) identical(x$id, id), state$updates)
    tail(calls, 1L)[[1L]]
  }
  state
}

test_that("multiline capability requires every terminal and layout condition", {
  supported <- list(num_chains = 2L, chain_format = NULL,
                    cli_version = package_version("3.6.6.9000"),
                    ansi = TRUE, multiline = TRUE, unsunk = TRUE)
  check <- function(args) do.call(nutpieR:::supports_multiline_progress, args)
  expect_true(check(supported))
  named_ansi <- supported
  named_ansi$ansi <- c(stdout = TRUE)
  expect_true(check(named_ansi))
  for (replacement in list(
    list(num_chains = 1L), list(chain_format = "{div} | {grad}"),
    list(cli_version = package_version("3.6.6")), list(ansi = FALSE),
    list(ansi = NA), list(multiline = FALSE), list(multiline = NA),
    list(unsunk = FALSE)
  )) {
    args <- supported
    args[names(replacement)] <- replacement
    expect_false(check(args))
  }
})

test_that("multiline updates independent chain bars through queued and mixed phases", {
  record <- multiline_recorder()
  ids <- c("header", "one", "two", "three")
  callback <- nutpieR:::make_multi_cli_progress_callback(
    3L, 100L, 100L, num_cores = 2L, ids = ids,
    update = record$update, done = record$finish
  )
  callback(multiline_snapshot())
  expect_setequal(vapply(record$updates, `[[`, "", "id"), ids)
  expect_equal(record$latest("header")$set, 120)
  expect_equal(record$latest("one")$set, 80)
  expect_equal(record$latest("two")$set, 40)
  expect_equal(record$latest("three")$set, 0)
  expect_identical(trimws(record$latest("one")$extra$phase), "warmup")
  expect_identical(trimws(record$latest("three")$extra$phase), "queued")
  expect_match(record$latest("one")$extra$grad, "- grad/draw", fixed = TRUE)

  mixed <- multiline_snapshot(c(120L, 90L, 0L), c(8280, 9000, 0),
                              tuning = c(FALSE, TRUE, TRUE))
  callback(mixed)
  expect_equal(record$latest("header")$set, 210)
  expect_identical(trimws(record$latest("one")$extra$phase), "sample")
  expect_identical(trimws(record$latest("two")$extra$phase), "warmup")
  expect_identical(trimws(record$latest("three")$extra$phase), "queued")
  extra <- record$latest("one")$extra
  expect_true(all(c("phase", "div", "grad", "step") %in% names(extra)))
  expect_match(as.character(extra$grad), "7.0", fixed = TRUE)
  expect_match(as.character(extra$step), "0.125", fixed = TRUE)
  expect_match(as.character(extra$div), "0", fixed = TRUE)

  final <- multiline_snapshot(rep(200L, 3L), c(8840, 9770, 1400),
                              tuning = rep(FALSE, 3L), started = rep(TRUE, 3L))
  callback(final)
  expect_equal(record$latest("header")$set, 600)
  for (id in ids[-1L]) {
    expect_equal(record$latest(id)$set, 200)
    expect_identical(trimws(record$latest(id)$extra$phase), "done")
  }
  expect_match(record$latest("one")$extra$grad, "7.0", fixed = TRUE)
  expect_match(record$latest("two")$extra$grad, "7.0", fixed = TRUE)
  nutpieR:::finish_progress_callback(callback, success = TRUE)
  nutpieR:::finish_progress_callback(callback, success = TRUE)
  expect_length(record$done, length(ids))
  expect_setequal(vapply(record$done, `[[`, "", "id"), ids)
  expect_true(all(vapply(record$done, function(x) identical(x$result, "done"), TRUE)))
})

test_that("multiline repaints only changed rows unless the width changes", {
  record <- multiline_recorder()
  ids <- c("header", "one", "two", "three")
  callback <- nutpieR:::make_multi_cli_progress_callback(
    3L, 100L, 100L, ids = ids, update = record$update, done = record$finish
  )
  withr::local_options(cli.width = 80L)
  snapshot <- multiline_snapshot()
  callback(snapshot)
  record$updates <- list()
  snapshot[[1L]]$runtime <- 10
  callback(snapshot)
  expect_length(record$updates, 0L)
  snapshot[[1L]]$finished_draws <- 81L
  callback(snapshot)
  expect_identical(vapply(record$updates, `[[`, "", "id"), ids[1:2])
  record$updates <- list()
  options(cli.width = 40L)
  callback(snapshot)
  expect_identical(vapply(record$updates, `[[`, "", "id"), ids)
})

test_that("multiline hints are pooled once and target the header", {
  record <- multiline_recorder()
  hint_ids <- character()
  testthat::local_mocked_bindings(
    emit_progress_hint = function(hints, level, msg) {
      hint_ids <<- c(hint_ids, hints$cli_bar_id)
      message(msg)
    },
    .package = "nutpieR"
  )
  callback <- nutpieR:::make_multi_cli_progress_callback(
    3L, 100L, 100L, num_cores = 2L, ids = c("header", "one", "two", "three"),
    update = record$update, done = record$finish
  )
  baseline <- multiline_snapshot(c(80L, 80L, 0L), c(8000, 8000, 0))
  expect_length(capture_messages(callback(baseline)), 0L)
  sample <- multiline_snapshot(c(110L, 110L, 0L), c(14300, 14300, 0),
                               tuning = c(FALSE, FALSE, TRUE),
                               divergences = c(1L, 2L, 0L))
  messages <- capture_messages(callback(sample))
  expect_equal(sum(grepl("divergent transitions detected", messages)), 1L)
  expect_equal(sum(grepl("210 gradient evaluations per draw", messages)), 1L)
  expect_equal(hint_ids, c("header", "header"))
  expect_length(capture_messages(callback(sample)), 0L)
  nutpieR:::finish_progress_callback(callback, success = FALSE)
})

test_that("multiline error and interrupt cleanup clears every bar exactly once", {
  for (kind in c("error", "interrupt")) {
    record <- multiline_recorder()
    callback <- nutpieR:::make_multi_cli_progress_callback(
      3L, 100L, 100L, ids = c("header", "one", "two", "three"),
      update = record$update, done = record$finish
    )
    tryCatch({
      callback(multiline_snapshot())
      if (kind == "error") stop("sampling failed")
      stop(structure(list(message = "interrupted"), class = c("interrupt", "condition")))
    }, error = function(e) NULL, interrupt = function(e) NULL,
    finally = nutpieR:::finish_progress_callback(callback, success = FALSE))
    nutpieR:::finish_progress_callback(callback, success = TRUE)
    expect_length(record$done, 4L)
    expect_setequal(vapply(record$done, `[[`, "", "id"), c("header", "one", "two", "three"))
    expect_true(all(vapply(record$done, function(x) identical(x$result, "clear"), TRUE)))
    expect_false(any(vapply(record$updates, function(x) identical(trimws(x$extra$phase), "done"), TRUE)))
  }
})

test_that("a failed multiline update disables updates and clears all IDs", {
  record <- multiline_recorder()
  n <- 0L
  callback <- nutpieR:::make_multi_cli_progress_callback(
    3L, 100L, 100L, ids = c("header", "one", "two", "three"),
    update = function(...) { n <<- n + 1L; stop("renderer failed") },
    done = record$finish
  )
  expect_silent(callback(multiline_snapshot()))
  calls <- n
  expect_silent(callback(multiline_snapshot(c(90L, 50L, 0L))))
  expect_equal(n, calls)
  nutpieR:::finish_progress_callback(callback, success = TRUE)
  nutpieR:::finish_progress_callback(callback, success = FALSE)
  expect_length(record$done, 4L)
  expect_setequal(vapply(record$done, `[[`, "", "id"), c("header", "one", "two", "three"))
  expect_true(all(vapply(record$done, function(x) identical(x$result, "clear"), TRUE)))
})

local_multiline_renderer <- function(.local_envir = parent.frame()) {
  skip_if(utils::packageVersion("cli") < "3.6.6.9000")
  testthat::local_mocked_bindings(
    progress_console_height = function() 24L,
    progress_messages_muffled = function() FALSE,
    .package = "nutpieR", .env = .local_envir
  )
  withr::local_options(list(
    cli.ansi = TRUE, cli.progress_multiline = TRUE,
    cli.progress_handlers = "cli", cli.progress_show_after = 0,
    cli.num_colors = 1L, cli.unicode = FALSE
  ), .local_envir = .local_envir)
}

multiline_render <- function(code) {
  # Multiline repaint frames can share a captured line, separated by CR.
  lines <- unlist(strsplit(cli::ansi_strip(cli::cli_fmt(code)), "[\r\n]"), use.names = FALSE)
  lines[nzchar(lines)]
}

expect_multiline_fits <- function(lines, width) {
  expect_gt(length(lines), 0L)
  expect_true(all(cli::ansi_nchar(lines, type = "width") < width))
}

test_that("double-digit chain labels keep the bars aligned", {
  local_multiline_renderer()
  withr::local_options(cli.width = 100L)
  callback <- nutpieR:::make_multi_cli_progress_callback(10L, 100L, 100L)
  on.exit(multiline_render(nutpieR:::finish_progress_callback(callback)), add = TRUE)
  snapshot <- multiline_snapshot(rep(80L, 10L), tuning = rep(TRUE, 10L),
                                 started = rep(TRUE, 10L), divergences = integer(10L))
  rows <- multiline_render(callback(snapshot))
  rows <- rows[grepl("^c(9|10) +warmup", rows)]
  expect_setequal(sub(" .*", "", rows), c("c9", "c10"))
  bars <- as.integer(regexpr("|", rows, fixed = TRUE))
  expect_true(all(bars > 0L))
  expect_length(unique(bars), 1L)
})

test_that("real multiline progress fits narrow, medium and wide terminals", {
  local_multiline_renderer()
  for (width in c(40L, 62L, 120L)) {
    withr::local_options(cli.width = width)
    callback <- nutpieR:::make_multi_cli_progress_callback(3L, 100L, 100L, num_cores = 2L)
    lines <- multiline_render(callback(multiline_snapshot()))
    expect_multiline_fits(lines, width)
    rendered <- paste(lines, collapse = "\n")
    expect_match(rendered, "queued", fixed = TRUE)
    expect_match(rendered, "warmup", fixed = TRUE)
    expect_match(rendered, "div: 0", fixed = TRUE)
    final <- multiline_render(nutpieR:::finish_progress_callback(callback, success = TRUE))
    expect_multiline_fits(final, width)
  }
})

multiline_final_rows <- function(lines) {
  # cli may repaint cached old-width rows before all bars receive an update.
  # Check the last rendering of each row, not those intermediate frames.
  vapply(c("Sampling", "c1", "c2", "c3"), function(name) {
    rows <- lines[grepl(paste0("^", name, " "), lines)]
    expect_gt(length(rows), 0L)
    tail(rows, 1L)
  }, "")
}

test_that("multiline progress final rows fit after unchanged snapshots resize", {
  local_multiline_renderer()
  withr::local_options(cli.width = 120L)
  callback <- nutpieR:::make_multi_cli_progress_callback(3L, 100L, 100L, num_cores = 2L)
  snapshot <- multiline_snapshot()
  expect_multiline_fits(multiline_render(callback(snapshot)), 120L)
  expect_length(multiline_render(callback(snapshot)), 0L)
  for (width in c(40L, 62L, 120L)) {
    options(cli.width = width)
    lines <- multiline_final_rows(multiline_render(callback(snapshot)))
    expect_multiline_fits(lines, width)
    expect_match(paste(lines, collapse = "\n"), "queued", fixed = TRUE)
  }
  options(cli.width = 40L)
  final <- multiline_final_rows(multiline_render(
    nutpieR:::finish_progress_callback(callback, success = TRUE)
  ))
  expect_multiline_fits(final, 40L)
})

test_that("sampling errors and interrupts clear multiline bars on exit", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  for (kind in c("error", "interrupt")) {
    record <- multiline_recorder()
    testthat::local_mocked_bindings(
      make_cli_progress_callback = function(...) {
        nutpieR:::make_multi_cli_progress_callback(
          2L, 10L, 10L, ids = c("header", "one", "two"),
          update = record$update, done = record$finish
        )
      },
      sample_stan = function(...) {
        if (kind == "error") stop("sampling failed")
        stop(structure(list(message = "interrupted"), class = c("interrupt", "condition")))
      },
      .package = "nutpieR"
    )
    condition <- tryCatch(
      nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
                    num_chains = 2L, num_warmup = 10L, num_draws = 10L,
                    seed = 1L, progress = "cli"),
      error = identity, interrupt = identity
    )
    expect_s3_class(condition, if (kind == "error") "error" else "interrupt")
    expect_length(record$done, 3L)
    expect_setequal(vapply(record$done, `[[`, "", "id"), c("header", "one", "two"))
    expect_true(all(vapply(record$done, function(x) identical(x$result, "clear"), TRUE)))
  }
})

test_that("multiline formatter rows fit each width during resize", {
  local_multiline_renderer()
  for (header in c(TRUE, FALSE)) {
    observed <- character()
    row <- function() {
      line <- nutpieR:::format_multi_cli_progress_line(header = header)
      observed <<- c(observed, line)
      line
    }
    id <- cli::cli_progress_bar(
      name = if (header) "Sampling" else "c1", total = 200L,
      type = "custom", format = "{row()}", format_done = "{row()}",
      extra = list(chains = "2 running, 1 queued, 0 done", phase = "warmup",
                   div = "div: 0", grad = "7.0 grad/draw", step = "0.125"),
      auto_terminate = FALSE, .auto_close = FALSE, current = FALSE
    )
    for (width in c(120L, 40L, 62L, 120L)) {
      withr::local_options(cli.width = width)
      observed <- character()
      invisible(multiline_render(cli::cli_progress_update(id = id, set = 80L, force = TRUE)))
      expect_multiline_fits(observed, width)
    }
    invisible(multiline_render(cli::cli_progress_done(id = id, result = "clear")))
  }
})

test_that("cli wrapper selects multiline only for the default supported layout", {
  capability <- nutpieR:::supports_multiline_progress
  testthat::local_mocked_bindings(
    supports_multiline_progress = function(num_chains, chain_format) capability(
      num_chains, chain_format, cli_version = package_version("3.6.6.9000"),
      ansi = TRUE, multiline = TRUE, unsunk = TRUE
    ),
    make_multi_cli_progress_callback = function(...) "multi",
    make_single_cli_progress_callback = function(...) "single",
    .package = "nutpieR"
  )
  expect_identical(nutpieR:::make_cli_progress_callback(2L, 10L, 10L), "multi")
  expect_identical(nutpieR:::make_cli_progress_callback(
    2L, 10L, 10L, chain_format = "{div} | {grad}"
  ), "single")
  expect_identical(nutpieR:::make_cli_progress_callback(1L, 10L, 10L), "single")
})

test_that("muted multiline callbacks create, update and finish no bars", {
  calls <- character()
  testthat::local_mocked_bindings(
    progress_messages_muffled = function() TRUE, .package = "nutpieR"
  )
  testthat::local_mocked_bindings(
    cli_progress_bar = function(...) { calls <<- c(calls, "create"); "bar" },
    .package = "cli"
  )
  callback <- nutpieR:::make_multi_cli_progress_callback(
    3L, 100L, 100L,
    update = function(...) calls <<- c(calls, "update"),
    done = function(...) calls <<- c(calls, "done")
  )
  callback(multiline_snapshot())
  nutpieR:::finish_progress_callback(callback, success = TRUE)
  nutpieR:::finish_progress_callback(callback, success = FALSE)
  expect_length(calls, 0L)
})
