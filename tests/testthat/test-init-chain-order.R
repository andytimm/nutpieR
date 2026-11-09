test_that("supplied starts map to output chains independently of worker count", {
  model <- try_compile("chain_init_order", code = "
    parameters {
      real marker;
      real z;
    }
    model {
      marker ~ normal(0, 1e100);
      z ~ std_normal();
    }
    generated quantities {
      real chain_marker = marker / 1e100;
    }
  ")
  skip_if(is.null(model), "Chain-init model not compiled")

  # A proper but very wide normal gives each chain an observable marker.
  # At these starts its floating-point spacing is about 1e84: ordinary HMC
  # steps cannot change it. One warmup draw avoids learning its large scale.
  # Unlike an ordinary parameter's first saved draw, chain_marker therefore
  # identifies the supplied start exactly, without a statistical comparison.
  sample_markers <- function(init, cores) {
    draws <- nutpie_sample(
      model, init = init, num_chains = 4, cores = cores,
      num_warmup = 1, num_draws = 3, max_treedepth = 2,
      seed = 712, progress = "none", refresh = 0
    )
    # Use base array subsetting, not posterior's dimension-preserving method.
    unname(unclass(draws)[, , "chain_marker"])
  }
  expect_markers <- function(actual, markers) {
    # Check shape separately so a malformed array reports its dimensions,
    # rather than failing inside waldo's matrix comparison.
    expect_identical(dim(actual), c(3L, 4L))
    expect_equal(as.numeric(actual), rep(markers, each = 3))
  }

  # Non-monotone starts expose swaps; both runs have more chains than workers.
  markers <- c(4, -2, 1, -3)
  # nuts-rs requires every initial gradient component to be nonzero.
  starts <- lapply(markers, function(marker) list(marker = marker * 1e100, z = 0.5))
  for (cores in c(1L, 2L)) {
    expect_markers(sample_markers(starts, cores), markers)
  }

  called <- integer()
  init <- function(chain_id) {
    called <<- c(called, chain_id)
    list(marker = markers[[chain_id]] * 1e100, z = 0.5)
  }
  expect_markers(sample_markers(init, cores = 2), markers)
  expect_identical(called, 1:4)

  # A complete named list resolves to one position, broadcast in Rust.
  expect_markers(
    sample_markers(list(marker = 2e100, z = 0.5), cores = 2),
    rep(2, 4)
  )

  # Partial starts get independent random fills for z without changing the
  # chain-indexed marker supplied by the caller.
  partial_starts <- lapply(markers, function(marker) list(marker = marker * 1e100))
  expect_markers(sample_markers(partial_starts, cores = 2), markers)

  paths <- vapply(seq_along(starts), function(i) tempfile(fileext = ".json"), character(1))
  on.exit(unlink(paths), add = TRUE)
  for (i in seq_along(starts)) {
    jsonlite::write_json(starts[[i]], paths[[i]], auto_unbox = TRUE, digits = NA)
  }
  expect_markers(sample_markers(paths, cores = 2), markers)
})
