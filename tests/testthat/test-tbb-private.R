# A compiled Stan model must load nutpieR's bundled TBB, even if a newer
# RcppParallel has already loaded a different TBB into this R process (#49).
test_that("macOS model links to private TBB and legacy paths resolve", {
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  model <- test_models$bernoulli
  expect_match(basename(model$lib_path), "_nutpieR_private_[0-9a-f]+\\.so$")
  raw <- sub("_nutpieR_private_[0-9a-f]+\\.so$", ".so", model$lib_path)
  expect_true(file.exists(raw))
  expect_identical(nutpieR:::private_tbb_model(raw), model$lib_path)
  legacy <- model
  legacy$lib_path <- raw
  expect_identical(nutpieR:::resolve_model(legacy), model$lib_path)

  deps <- suppressWarnings(system2("otool", c("-L", shQuote(model$lib_path)),
                                   stdout = TRUE))
  expect_false(any(grepl("@rpath/libtbb", deps, fixed = TRUE)))
  expect_true(any(grepl("@rpath/n_t[0-9a-f]+\\.dylib", deps)))
  load_commands <- suppressWarnings(system2("otool", c("-l", shQuote(model$lib_path)), stdout = TRUE))
  expect_true(any(grepl("path @loader_path (offset", load_commands, fixed = TRUE)))
  links <- list.files(dirname(model$lib_path),
                      pattern = "^n_[tmp][0-9a-f]+\\.dylib$", full.names = TRUE)
  expect_gte(length(links), 3L)
  expect_true(all(nzchar(Sys.readlink(links))))
  expect_true(all(file.exists(links)))
})

test_that("macOS proxy-free models keep private TBB without loading a proxy", {
  skip_on_os(c("windows", "linux", "solaris"))
  model <- nutpie_compile_model(
    code = "parameters { real x; } model { x ~ normal(0, 1); }",
    cache = FALSE, compile_args = "TBB_LIBRARIES=tbb", verbose = 0L
  )
  deps <- suppressWarnings(system2("otool", c("-L", shQuote(model$lib_path)),
                                   stdout = TRUE))
  expect_true(any(grepl("@rpath/n_t[0-9a-f]+\\.dylib", deps)))
  expect_false(any(grepl("@rpath/libtbb", deps, fixed = TRUE)))
  expect_false(any(grepl("@rpath/n_p[0-9a-f]+\\.dylib", deps)))
  draws <- nutpie_sample(model, num_draws = 5L, num_warmup = 5L,
                         num_chains = 1L, refresh = 0L)
  expect_equal(dim(draws)[[1L]], 5L)
})

test_that("macOS model TBB rpath may be relative to its library", {
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  raw <- sub("_nutpieR_private_[0-9a-f]+\\.so$", ".so",
             test_models$bernoulli$lib_path)
  paths <- suppressWarnings(system2("otool", c("-l", shQuote(raw)), stdout = TRUE))
  paths <- sub("^.*path (.+) \\(offset [0-9]+\\)$", "\\1",
               grep("path .+ \\(offset [0-9]+\\)$", paths, value = TRUE))
  tbb <- paths[file.exists(file.path(paths, "libtbb.dylib"))][[1L]]

  dir <- tempfile("nutpieR-relative-tbb-")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  copy <- file.path(dir, "relative_model.so")
  expect_true(file.copy(raw, copy))
  expect_true(file.symlink(tbb, file.path(dir, "tbb")))
  expect_equal(system2("install_name_tool",
                       c("-rpath", shQuote(tbb), "@loader_path/tbb", shQuote(copy))), 0L)
  expect_equal(system2("codesign", c("--force", "--sign", "-", shQuote(copy)),
                       stdout = FALSE, stderr = FALSE), 0L)
  private <- nutpieR:::private_tbb_model(copy)
  expect_true(file.exists(private))
  expect_false(any(grepl("@rpath/libtbb", suppressWarnings(
    system2("otool", c("-L", shQuote(private)), stdout = TRUE)), fixed = TRUE)))
})

test_that("macOS rejects unrecognized Stan TBB dylib names", {
  skip_on_os(c("windows", "linux", "solaris"))
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  raw <- sub("_nutpieR_private_[0-9a-f]+\\.so$", ".so",
             test_models$bernoulli$lib_path)
  copy <- tempfile("nutpieR-unknown-tbb-", fileext = ".so")
  on.exit(unlink(copy), add = TRUE)
  expect_true(file.copy(raw, copy))
  expect_equal(system2("install_name_tool", c("-change", "@rpath/libtbb.dylib",
                                               "@rpath/libtbb.12.dylib", shQuote(copy))), 0L)
  expect_error(nutpieR:::private_tbb_model(copy), "unsupported Stan TBB dependency")
})

test_that("macOS private TBB rpath fits with a long home directory", {
  skip_on_os(c("windows", "linux", "solaris"))
  model <- nutpie_compile_model(
    code = "parameters { real x; } model { x ~ normal(0, 1); }",
    cache = FALSE, compile_args = "TBB_LIBRARIES=tbb", verbose = 0L
  )
  raw <- sub("_nutpieR_private_[0-9a-f]+\\.so$", ".so", model$lib_path)
  dir <- tempfile("nutpieR-long-home-")
  long_home <- file.path(dir, paste(rep("long-home-component", 9L), collapse = "-"))
  dir.create(long_home, recursive = TRUE)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  copy <- file.path(dir, "model.so")
  expect_true(file.copy(raw, copy))

  withr::local_envvar(HOME = long_home)
  private <- nutpieR:::private_tbb_model(copy)
  expect_true(file.exists(private))
  expect_true(any(grepl("path @loader_path (offset", suppressWarnings(
    system2("otool", c("-l", shQuote(private)), stdout = TRUE)), fixed = TRUE)))
  links <- list.files(dir, pattern = "^n_t[0-9a-f]+\\.dylib$", full.names = TRUE)
  expect_length(links, 1L)
  expect_true(file.exists(links))
  unlink(dirname(Sys.readlink(links[[1L]])), recursive = TRUE)
  expect_false(file.exists(links))
  expect_identical(nutpieR:::private_tbb_model(private), private)
  expect_true(file.exists(links))

  # A partly deleted bundle must fail with guidance rather than silently use
  # the remaining files. Removing the whole bundle allows safe reconstruction.
  target <- Sys.readlink(links[[1L]])
  unlink(target)
  expect_error(nutpieR:::private_tbb_model(private), "incomplete private TBB bundle")
  unlink(dirname(target), recursive = TRUE)
  expect_identical(nutpieR:::private_tbb_model(private), private)

  # A broken link from another process must not be replaced by this build.
  unlink(links[[1L]])
  wrong <- file.path(dir, "other-tbb.dylib")
  expect_true(file.symlink(wrong, links[[1L]]))
  expect_error(nutpieR:::private_tbb_model(copy), "already exists")
  expect_identical(Sys.readlink(links[[1L]]), wrong)
})
