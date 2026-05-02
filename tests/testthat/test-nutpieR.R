# Synthetic name vectors mirroring the tp_gq Stan model:
#   parameters { mu; sigma; }            -> block
#   transformed parameters { mu_sq; }    -> block_tp adds mu_sq
#   generated quantities { y_rep[N=5]; } -> full adds y_rep
TP_GQ_BLOCK    <- c("mu", "sigma")
TP_GQ_BLOCK_TP <- c("mu", "sigma", "mu_sq")
TP_GQ_FULL_DOT <- c("mu", "sigma", "mu_sq",
                    "y_rep.1", "y_rep.2", "y_rep.3", "y_rep.4", "y_rep.5")
TP_GQ_FULL     <- nutpieR:::block_prefixes(TP_GQ_FULL_DOT)

# --- resolve_data unit tests -------------------------------------------------

test_that("resolve_data handles NULL", {
  expect_equal(nutpieR:::resolve_data(NULL), "")
})

test_that("resolve_data handles JSON string", {
  json <- '{"N": 10}'
  expect_equal(nutpieR:::resolve_data(json), json)
})

test_that("resolve_data handles list", {
  result <- nutpieR:::resolve_data(list(N = 10, y = c(0, 1)))
  parsed <- jsonlite::fromJSON(result)
  expect_equal(parsed$N, 10)
  expect_equal(parsed$y, c(0, 1))
})

test_that("resolve_data handles .json file", {
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))
  writeLines('{"N": 5}', tmp)
  result <- nutpieR:::resolve_data(tmp)
  expect_equal(jsonlite::fromJSON(result)$N, 5)
})

test_that("resolve_data rejects invalid input", {
  expect_error(nutpieR:::resolve_data(123))
})

# --- check_count unit tests --------------------------------------------------

test_that("check_count rejects malformed counts", {
  expect_error(nutpieR:::check_count(-1L, "num_draws", min = 1L), "num_draws")
  expect_error(nutpieR:::check_count(NA_integer_, "num_chains"), "num_chains")
  expect_error(nutpieR:::check_count(1.5, "num_warmup"), "num_warmup")
  expect_error(nutpieR:::check_count(c(1L, 2L), "num_chains"), "num_chains")
})

test_that("check_count enforces optional max", {
  expect_error(
    nutpieR:::check_count(10L, "seed", min = 0L, max = 5L),
    "must be <= 5"
  )
  expect_equal(nutpieR:::check_count(5L, "seed", min = 0L, max = 5L), 5L)
})

test_that("nutpie_sample rejects malformed seed", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  for (bad in list(NA_integer_, -1L, 1.5, 2^32)) {
    expect_error(
      nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
                    seed = bad, num_draws = 10, num_chains = 1,
                    refresh = 0),
      "seed"
    )
  }
})

test_that("nutpie_sample rejects malformed target_accept", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  expect_error(
    nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
                  target_accept = 1.5),
    "target_accept"
  )
})

test_that("cores defaults to 1 when detectCores returns NA", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  testthat::local_mocked_bindings(
    detectCores = function(...) NA_integer_,
    .package = "parallel"
  )
  draws <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
                         num_draws = 50, num_warmup = 50, num_chains = 1,
                         seed = 1L, refresh = 0)
  expect_s3_class(draws, "draws_array")
})

test_that("low_rank_modified_mass_matrix is deprecated but still works", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  expect_warning(
    draws <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
                           num_draws = 50, num_warmup = 50, num_chains = 1,
                           seed = 1L, refresh = 0,
                           low_rank_modified_mass_matrix = TRUE),
    "deprecated"
  )
  expect_s3_class(draws, "draws_array")
  cfg <- jsonlite::fromJSON(attr(draws, "sampler_config"))
  # low_rank settings have a `mass_matrix_update_freq` of 20 (vs. 1 for diag).
  expect_equal(cfg$adapt_options$mass_matrix_update_freq, 20)
})

test_that("adaptation = 'low_rank' matches the deprecated flag's behaviour", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  draws_new <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
                             num_draws = 100, num_warmup = 100, num_chains = 1,
                             seed = 1L, refresh = 0,
                             adaptation = "low_rank")
  expect_warning(
    draws_old <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
                               num_draws = 100, num_warmup = 100,
                               num_chains = 1, seed = 1L, refresh = 0,
                               low_rank_modified_mass_matrix = TRUE),
    "deprecated"
  )
  # Same seed + same path should yield identical draws.
  expect_equal(as.numeric(draws_new), as.numeric(draws_old))
})

# --- resolve_constrain_flags_impl unit tests (no handle) ---------------------

test_that("resolve_constrain_flags short-circuits without touching the handle when pars is NULL", {
  # handle = NULL would crash if the wrapper evaluated bs_*_names on it.
  flags <- nutpieR:::resolve_constrain_flags(handle = NULL, pars = NULL,
                                             include = TRUE)
  expect_true(flags$include_tp)
  expect_true(flags$include_gq)
})

test_that("resolve_constrain_flags drops both flags when only block kept", {
  flags <- nutpieR:::resolve_constrain_flags_impl(
    block = TP_GQ_BLOCK, block_tp = TP_GQ_BLOCK_TP, full = TP_GQ_FULL,
    pars = c("mu", "sigma"), include = TRUE
  )
  expect_false(flags$include_tp)
  expect_false(flags$include_gq)
})

test_that("resolve_constrain_flags keeps include_tp when TP kept, no GQ", {
  flags <- nutpieR:::resolve_constrain_flags_impl(
    block = TP_GQ_BLOCK, block_tp = TP_GQ_BLOCK_TP, full = TP_GQ_FULL,
    pars = "mu_sq", include = TRUE
  )
  expect_true(flags$include_tp)
  expect_false(flags$include_gq)
})

test_that("resolve_constrain_flags forces include_tp when GQ kept (conservative)", {
  flags <- nutpieR:::resolve_constrain_flags_impl(
    block = TP_GQ_BLOCK, block_tp = TP_GQ_BLOCK_TP, full = TP_GQ_FULL,
    pars = "y_rep", include = TRUE
  )
  expect_true(flags$include_tp)
  expect_true(flags$include_gq)
})

test_that("resolve_constrain_flags blacklist excluding GQ name drops include_gq", {
  flags <- nutpieR:::resolve_constrain_flags_impl(
    block = TP_GQ_BLOCK, block_tp = TP_GQ_BLOCK_TP, full = TP_GQ_FULL,
    pars = "y_rep", include = FALSE
  )
  # mu_sq (TP) is still kept, so include_tp must remain TRUE
  expect_true(flags$include_tp)
  expect_false(flags$include_gq)
})

test_that("resolve_constrain_flags errors on unknown parameter names", {
  expect_error(
    nutpieR:::resolve_constrain_flags_impl(
      block = TP_GQ_BLOCK, block_tp = TP_GQ_BLOCK_TP, full = TP_GQ_FULL,
      pars = "nonexistent", include = TRUE
    ),
    "Unknown parameter"
  )
})

# --- resolve_keep_indices unit tests -----------------------------------------

test_that("resolve_keep_indices returns NULL when pars is NULL", {
  expect_null(nutpieR:::resolve_keep_indices(TP_GQ_FULL_DOT, NULL, TRUE))
})

test_that("resolve_keep_indices whitelist returns matching column indices", {
  idx <- nutpieR:::resolve_keep_indices(TP_GQ_FULL_DOT, "y_rep", TRUE)
  # 0-indexed column positions for y_rep.1..y_rep.5 in TP_GQ_FULL_DOT
  expect_equal(idx, 3:7)
})

test_that("resolve_keep_indices blacklist excludes matching", {
  idx <- nutpieR:::resolve_keep_indices(TP_GQ_FULL_DOT, "y_rep", FALSE)
  # Everything except y_rep.*
  expect_equal(idx, 0:2)
})

test_that("resolve_keep_indices empty whitelist errors", {
  # pars = character(0) with include = TRUE means "keep nothing", not
  # "keep everything" — guards against programmatic selections like
  # pars = intersect(user_pars, available) that come up empty.
  expect_error(
    nutpieR:::resolve_keep_indices(TP_GQ_FULL_DOT, character(0), TRUE),
    "remove all variables"
  )
})

test_that("resolve_keep_indices full exclusion errors", {
  expect_error(
    nutpieR:::resolve_keep_indices(c("mu", "sigma"),
                                   c("mu", "sigma"), FALSE),
    "remove all variables"
  )
})

test_that("resolve_keep_indices empty blacklist returns NULL (keep all)", {
  expect_null(nutpieR:::resolve_keep_indices(TP_GQ_FULL_DOT, character(0), FALSE))
})

# --- Compilation -------------------------------------------------------------

test_that("nutpie_compile_model returns nutpie_model", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  expect_s3_class(test_models$bernoulli, "nutpie_model")
  expect_true(file.exists(test_models$bernoulli$lib_path))
})

test_that("nutpie_compile_model accepts code string", {
  skip_if(is.null(test_models$code_string), "Code string model not compiled")
  expect_s3_class(test_models$code_string, "nutpie_model")
  expect_true(file.exists(test_models$code_string$lib_path))
})

test_that("nutpie_compile_model rejects both stan_file and code", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  expect_error(
    nutpie_compile_model(stan_file = stan_file, code = "data {}"),
    "exactly one"
  )
})

test_that("nutpie_compile_model rejects neither stan_file nor code", {
  expect_error(nutpie_compile_model(), "exactly one")
})

# --- Canonical bernoulli end-to-end ------------------------------------------
# Folds: dims, posterior mean, num_warmup/num_draws attrs, max_treedepth +
# target_accept past-validation, must_have diagnostic fields, type promises,
# logp non-zero, list-column drops, nutpie_warmup_draws-errors-without-save.

test_that("bernoulli end-to-end run surfaces draws + diagnostics + attributes", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
    num_draws = 200, num_warmup = 200, num_chains = 2, seed = 42, refresh = 0,
    max_treedepth = 8, target_accept = 0.9
  )

  # Shape
  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::ndraws(draws), 400)
  expect_equal(posterior::nchains(draws), 2)
  expect_equal(posterior::niterations(draws), 200)
  expect_true("theta" %in% posterior::variables(draws))

  # Attributes record sampling configuration
  expect_equal(attr(draws, "num_warmup"), 200L)
  expect_equal(attr(draws, "num_draws"), 200L)

  # Posterior mean: 2/10 successes + Beta(1,1) prior -> Beta(3,9), mean = 0.25
  summ <- posterior::summarize_draws(draws)
  theta_mean <- summ$mean[summ$variable == "theta"]
  expect_true(abs(theta_mean - 0.25) < 0.05)

  # Diagnostics: load-bearing fields with promised types
  diag <- nutpie_diagnostics(draws)
  must_have <- c("diverging", "depth", "logp", "energy", "n_steps",
                 "step_size_bar", "mean_tree_accept")
  for (field in must_have) {
    expect_true(field %in% names(diag), info = paste("missing:", field))
  }
  expect_type(diag$diverging, "logical")
  expect_type(diag$depth, "integer")
  expect_type(diag$n_steps, "integer")
  expect_type(diag$logp, "double")
  expect_type(diag$energy, "double")
  expect_true(all(diag$mean_tree_accept >= 0 & diag$mean_tree_accept <= 1))

  # logp must be real floats — guards a class of bug where a renamed/missing
  # column would silently leave the zero-init buffer in place (printing as
  # integers).
  expect_true(any(diag$logp != 0))
  expect_true(any(abs(diag$logp - round(diag$logp)) > 1e-8))

  # Default flags drop their list columns
  expect_false("divergence_start" %in% names(diag))
  expect_false("mass_matrix_inv" %in% names(diag))
  expect_false("unconstrained_draw" %in% names(diag))
  expect_false("gradient" %in% names(diag))

  # save_warmup wasn't passed -> nutpie_warmup_draws should error
  expect_error(nutpie_warmup_draws(draws), "save_warmup")
})

# --- save_warmup integration -------------------------------------------------

test_that("save_warmup returns warmup draws + diagnostics with tuning field", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
    num_draws = 100, num_warmup = 50, num_chains = 2, seed = 42,
    refresh = 0, save_warmup = TRUE
  )

  warmup <- nutpie_warmup_draws(draws)
  expect_s3_class(warmup, "draws_array")
  expect_equal(posterior::niterations(warmup), 50)
  expect_equal(posterior::nchains(warmup), 2)
  expect_true("theta" %in% posterior::variables(warmup))

  warmup_diag <- nutpie_warmup_diagnostics(draws)
  expect_type(warmup_diag, "list")
  expect_equal(length(warmup_diag$diverging), 100) # 50 * 2 chains
  expect_true("logp" %in% names(warmup_diag))
  expect_true("tuning" %in% names(warmup_diag))
  expect_type(warmup_diag$logp, "double")
  expect_true(any(warmup_diag$logp != 0))
  expect_true(any(warmup_diag$tuning))
})

# --- pars whitelist + warmup filtering on normal -----------------------------

test_that("pars whitelist + save_warmup filters both draws and warmup", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(test_models$normal, data = normal_data(),
    num_draws = 50, num_warmup = 50, num_chains = 1, seed = 42,
    refresh = 0, save_warmup = TRUE, pars = "sigma"
  )

  expect_equal(posterior::variables(draws), "sigma")
  warmup <- nutpie_warmup_draws(draws)
  expect_equal(posterior::variables(warmup), "sigma")
})

# --- tp_gq pars flag-path coverage -------------------------------------------
# Three sampler runs, three (include_tp, include_gq) paths.

test_that("tp_gq pars whitelist exercises all three flag paths", {
  skip_if(is.null(test_models$tp_gq), "tp_gq model not compiled")

  # block-only -> (FALSE, FALSE): mu_sq (TP) and y_rep (GQ) must not appear
  draws_block <- nutpie_sample(test_models$tp_gq, data = normal_data(),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    pars = c("mu", "sigma")
  )
  vars_block <- posterior::variables(draws_block)
  expect_setequal(vars_block, c("mu", "sigma"))
  expect_false("mu_sq" %in% vars_block)
  expect_false(any(grepl("^y_rep", vars_block)))

  # TP-only -> (TRUE, FALSE)
  draws_tp <- nutpie_sample(test_models$tp_gq, data = normal_data(),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    pars = "mu_sq"
  )
  vars_tp <- posterior::variables(draws_tp)
  expect_equal(vars_tp, "mu_sq")
  expect_false(any(grepl("^y_rep", vars_tp)))

  # GQ -> (TRUE, TRUE), TP forced on internally
  draws_gq <- nutpie_sample(test_models$tp_gq, data = normal_data(),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    pars = "y_rep"
  )
  vars_gq <- posterior::variables(draws_gq)
  expect_length(vars_gq, 5L)  # y_rep is length-N=5
  expect_true(all(grepl("^y_rep", vars_gq)))
  expect_false("mu" %in% vars_gq)
  expect_false("mu_sq" %in% vars_gq)
})

# --- pars on a model with no TP/GQ -------------------------------------------

test_that("bernoulli pars = 'theta' exercises the (FALSE, FALSE) flag path", {
  # bernoulli has only the `theta` block parameter — no TP, no GQ.
  # Confirms the flag-path is a no-op when include_tp / include_gq are
  # already FALSE for the model itself.
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    pars = "theta"
  )
  expect_equal(posterior::variables(draws), "theta")
})

# --- store_* flag surfacing --------------------------------------------------

test_that("store_unconstrained / gradient / mass_matrix surface their columns", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  num_draws <- 50
  draws <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
    num_draws = num_draws, num_chains = 1, seed = 42, refresh = 0,
    store_unconstrained = TRUE, store_gradient = TRUE, store_mass_matrix = TRUE
  )
  diag <- nutpie_diagnostics(draws)

  expect_true("unconstrained_draw" %in% names(diag))
  expect_true("gradient" %in% names(diag))
  expect_true("mass_matrix_inv" %in% names(diag))

  # Bernoulli has 1 unconstrained parameter, so each list-of-vectors column
  # collapses to a (num_draws, 1) numeric matrix via the uniform-width path.
  for (col in c("unconstrained_draw", "gradient", "mass_matrix_inv")) {
    expect_true(is.matrix(diag[[col]]), info = col)
    expect_type(diag[[col]], "double")
    expect_equal(dim(diag[[col]]), c(num_draws, 1L), info = col)
  }
})

# --- store_divergences detail (conditional) ----------------------------------

test_that("store_divergences exposes divergence detail when divergences occur", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  # Force divergences via tiny max_treedepth + low target_accept on a model
  # that's easy to push into pathology. Conditional on actually getting some.
  draws <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
    num_draws = 200, num_chains = 4, seed = 42, refresh = 0,
    store_divergences = TRUE, max_treedepth = 1, target_accept = 0.5
  )
  diag <- nutpie_diagnostics(draws)
  skip_if_not(sum(diag$diverging) > 0, "no divergences in this run")

  expect_true("divergence_start" %in% names(diag))
  expect_true("divergence_end" %in% names(diag))
  expect_true("divergence_momentum" %in% names(diag))
  expect_type(diag$divergence_start, "list")
  expect_equal(length(diag$divergence_start), length(diag$diverging))
  div_idx <- which(diag$diverging)
  expect_true(all(!vapply(diag$divergence_start[div_idx], is.null, logical(1))))
})

# --- low-rank mass matrix ----------------------------------------------------

test_that("adaptation = 'low_rank' produces valid draws", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  # store_mass_matrix is requested too: low-rank's mass matrix has a
  # different internal representation than the diagonal one and may not
  # surface `mass_matrix_inv`, but the run must complete cleanly with
  # logp populated.
  draws <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
    num_draws = 100, num_chains = 2, seed = 42, refresh = 0,
    adaptation = "low_rank", store_mass_matrix = TRUE
  )
  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::niterations(draws), 100)
  expect_true("theta" %in% posterior::variables(draws))

  # Should reach a similar posterior as standard mass matrix
  summ <- posterior::summarize_draws(draws)
  theta_mean <- summ$mean[summ$variable == "theta"]
  expect_true(abs(theta_mean - 0.25) < 0.1)

  # logp populated — guards a class of bug where extraction silently leaves
  # a zero-init buffer in place
  diag <- nutpie_diagnostics(draws)
  expect_type(diag$logp, "double")
  expect_true(any(diag$logp != 0))
})

# --- bad-data error path -----------------------------------------------------

test_that("sampling with bad data gives R error, not crash", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  expect_error(
    nutpie_sample(test_models$bernoulli, data = '{"bad": "json format"}',
                  num_draws = 10, num_chains = 1, seed = 42, refresh = 0)
  )
})

# --- sampler_config attribute ------------------------------------------------

test_that("sampler_config is parseable JSON capturing effective settings", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  draws <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
                         num_draws = 50, num_warmup = 50, num_chains = 1,
                         seed = 1L, refresh = 0, target_accept = 0.9,
                         max_treedepth = 8L, extra_doublings = 2L)
  cfg_str <- attr(draws, "sampler_config")
  expect_type(cfg_str, "character")
  expect_true(nzchar(cfg_str))
  cfg <- jsonlite::fromJSON(cfg_str)
  expect_equal(cfg$num_tune, 50)
  expect_equal(cfg$num_draws, 50)
  expect_equal(cfg$maxdepth, 8)
  expect_equal(cfg$extra_doublings, 2)
  expect_equal(cfg$adapt_options$step_size_settings$target_accept, 0.9)
})

test_that("unspecified target_accept / max_treedepth use nuts-rs defaults", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  draws <- nutpie_sample(test_models$bernoulli, data = bernoulli_data(),
                         num_draws = 50, num_warmup = 50, num_chains = 1,
                         seed = 1L, refresh = 0)
  expect_s3_class(draws, "draws_array")
  cfg <- jsonlite::fromJSON(attr(draws, "sampler_config"))
  # nuts-rs DiagGradNutsSettings defaults: maxdepth = 10, target_accept = 0.8.
  expect_equal(cfg$maxdepth, 10)
  expect_equal(cfg$adapt_options$step_size_settings$target_accept, 0.8)
})

# --- mass_matrix_inv shape ---------------------------------------------------

test_that("store_mass_matrix surfaces mass_matrix_inv as a numeric matrix", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")
  num_draws <- 60
  num_chains <- 2
  draws <- nutpie_sample(test_models$normal, data = normal_data(),
                         num_draws = num_draws, num_warmup = 80,
                         num_chains = num_chains, seed = 1L, refresh = 0,
                         store_mass_matrix = TRUE)
  diag <- nutpie_diagnostics(draws)
  mm <- diag$mass_matrix_inv
  expect_true(is.matrix(mm))
  expect_type(mm, "double")
  ndim_unc <- length(nutpie_param_names(test_models$normal,
                                        data = normal_data(),
                                        which = "unconstrained"))
  expect_equal(dim(mm), c(num_draws * num_chains, ndim_unc))
})

# --- diagnostics on a non-nutpie object --------------------------------------

test_that("nutpie_diagnostics errors on non-nutpie object", {
  expect_error(nutpie_diagnostics(1:10), "No diagnostics found")
})
