progress_width_snapshot <- function() {
  lapply(seq_len(2L), function(chain) list(
    chain = chain, finished_draws = 50L, total_draws = 100L,
    divergences = 0L, tuning = FALSE, started = TRUE,
    latest_num_steps = 7L, total_num_steps = 350,
    step_size = 0.5, runtime = 1, divergent_draws = integer()
  ))
}

capture_progress_lines <- function(code) {
  lines <- cli::ansi_strip(cli::cli_fmt(code))
  lines[nzchar(lines)]
}

local_progress_width_options <- function(.local_envir = parent.frame()) {
  # testthat muffles messages; enable callbacks while testing the real renderer.
  testthat::local_mocked_bindings(
    progress_messages_muffled = function() FALSE,
    .package = "nutpieR", .env = .local_envir
  )
  withr::local_options(list(
    cli.progress_handlers = "cli", cli.progress_show_after = 0,
    cli.num_colors = 1L, cli.unicode = FALSE
  ), .local_envir = .local_envir)
}

expect_progress_fits <- function(lines, width) {
  expect_gt(length(lines), 0L)
  # Leave the last terminal cell free: cli clears it when redrawing the line.
  expect_true(all(cli::ansi_nchar(lines, type = "width") < width))
}

test_that("real cli progress keeps divergence counts visible at console widths", {
  local_progress_width_options()
  for (width in c(40L, 62L, 120L)) {
    withr::local_options(cli.width = width)
    # Construct outside rendering capture: callbacks remember message muting.
    callback <- nutpieR:::make_cli_progress_callback(2L, 0L, 100L)
    lines <- capture_progress_lines(callback(progress_width_snapshot()))
    expect_progress_fits(lines, width)
    expect_match(paste(lines, collapse = ""), "div: 0", fixed = TRUE)
    if (width == 120L) {
      expect_match(paste(lines, collapse = ""), "grad/draw", fixed = TRUE)
    }
    final <- capture_progress_lines(nutpieR:::finish_progress_callback(callback))
    expect_progress_fits(final, width)
    expect_match(paste(final, collapse = ""), "div: 0", fixed = TRUE)
  }
})

test_that("cli compacts surrounding layout without dropping custom status that fits", {
  local_progress_width_options()
  for (width in c(62L, 120L)) {
    withr::local_options(cli.width = width)
    callback <- nutpieR:::make_cli_progress_callback(
      2L, 0L, 100L, chain_format = "geometry {div} | {tdepth}"
    )
    lines <- capture_progress_lines(callback(progress_width_snapshot()))
    expect_progress_fits(lines, width)
    expected <- nutpieR:::format_status_tokens(
      progress_width_snapshot(),
      nutpieR:::summarize_progress_snapshot(progress_width_snapshot()),
      "geometry {div} | {tdepth}"
    )
    expect_match(paste(lines, collapse = ""), expected, fixed = TRUE)
    final <- capture_progress_lines(nutpieR:::finish_progress_callback(callback))
    expect_progress_fits(final, width)
    expect_match(paste(final, collapse = ""), expected, fixed = TRUE)
  }
})

test_that("one cli callback redraws unchanged snapshots on resize and fits at finish", {
  local_progress_width_options()
  withr::local_options(cli.width = 120L)
  callback <- nutpieR:::make_cli_progress_callback(2L, 0L, 100L)
  snapshot <- progress_width_snapshot()
  wide <- capture_progress_lines(callback(snapshot))
  expect_progress_fits(wide, 120L)
  expect_length(capture_progress_lines(callback(snapshot)), 0L)

  options(cli.width = 40L)
  narrow <- capture_progress_lines(callback(snapshot))
  expect_progress_fits(narrow, 40L)
  expect_match(paste(narrow, collapse = ""), "div: 0", fixed = TRUE)

  options(cli.width = 120L)
  restored <- capture_progress_lines(callback(snapshot))
  expect_progress_fits(restored, 120L)
  expect_match(paste(restored, collapse = ""), "grad/draw", fixed = TRUE)

  # A final resize need not receive another sampler snapshot.
  options(cli.width = 40L)
  final <- capture_progress_lines(nutpieR:::finish_progress_callback(callback))
  expect_progress_fits(final, 40L)
  expect_match(paste(final, collapse = ""), "div: 0", fixed = TRUE)
})

test_that("warmup progress keeps the native bar and divergences at the 62-column boundary", {
  local_progress_width_options()
  withr::local_options(cli.width = 62L)
  snapshot <- lapply(seq_len(6L), function(chain) list(
    chain = chain, finished_draws = if (chain <= 2L) 808L else 0L,
    total_draws = 2000L, divergences = 0L, tuning = TRUE,
    started = chain <= 2L, latest_num_steps = 7L,
    total_num_steps = if (chain <= 2L) 5494 else 0,
    step_size = 0.5, runtime = 1, divergent_draws = integer()
  ))
  callback <- nutpieR:::make_cli_progress_callback(6L, 1000L, 1000L)
  lines <- capture_progress_lines(callback(snapshot))
  expect_progress_fits(lines, 62L)
  rendered <- paste(lines, collapse = "")
  expect_match(rendered, "warmup", fixed = TRUE)
  expect_match(rendered, "\\|[=>-]+ \\|")
  expect_match(rendered, "div: 0", fixed = TRUE)
  final <- capture_progress_lines(nutpieR:::finish_progress_callback(callback))
  expect_progress_fits(final, 62L)
})
