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
  expect_true(any(grepl("/nutpieR/tbb/", load_commands, fixed = TRUE)))
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
