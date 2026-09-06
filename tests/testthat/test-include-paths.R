test_that("include path normalization preserves flag and path order", {
  flags <- c("--O1", "--include-paths=C:\\first,D:\\second/sub",
             "--include-paths=C:\\third", "--name=keep\\this")
  expected <- c("--O1", "--include-paths=C:/first,D:/second/sub",
                "--include-paths=C:/third", "--name=keep\\this")
  expect_identical(
    nutpieR:::normalize_stanc_include_paths(flags, windows = TRUE), expected
  )
  expect_identical(
    nutpieR:::normalize_stanc_include_paths(flags, windows = FALSE), flags
  )
  expect_identical(
    nutpieR:::normalize_stanc_include_paths(expected, windows = TRUE), expected
  )
  expect_identical(
    nutpieR:::normalize_stanc_include_paths(character(), windows = TRUE), character()
  )
})

test_that("normalizing include paths keeps shell-sensitive arguments untrackable", {
  testthat::local_mocked_bindings(stanc_make_override_present = function() FALSE,
                                .package = "nutpieR")
  normalize <- function(x) nutpieR:::normalize_stanc_include_paths(x, windows = TRUE)
  expect_true(nutpieR:::stanc_tracking_supported(normalize("--include-paths=C:\\inc")))
  for (flag in c("--include-paths=C:\\has space", "--include-paths=C:\\$HOME",
                 "--name=keep\\this", "--auto-format")) {
    expect_false(nutpieR:::stanc_tracking_supported(normalize(flag)))
  }
})

test_that("Windows include flags reach tracking and compilation normalized", {
  skip_on_os(c("linux", "mac", "solaris"))
  seen <- new.env(parent = emptyenv())
  testthat::local_mocked_bindings(
    bundle_for_compile = function(stan_file, code, stanc_args, compile_args) {
      seen$tracking <- stanc_args
      nutpieR:::inline_bundle(code)
    },
    compile_no_cache = function(bundle, stanc_args, compile_args, verbose) {
      seen$compilation <- stanc_args
      NULL
    },
    .package = "nutpieR"
  )
  nutpie_compile_model(code = "parameters { real x; } model {}", cache = FALSE,
                       stanc_args = "--include-paths=C:\\inc,D:\\other")
  expect_identical(seen$tracking, "--include-paths=C:/inc,D:/other")
  expect_identical(seen$compilation, seen$tracking)
})
