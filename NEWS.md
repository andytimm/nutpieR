# nutpieR 1.7.2

* Sampling output now closes with a per-chain finish summary and, when
  any divergences occurred, a final warning matching cmdstanr's format
  (with the canonical mc-stan.org/misc/warnings link).

# nutpieR 1.7.1

* Arrow `Utf8` diagnostic columns (e.g. `divergence_message`) now
  surface as character vectors instead of being skipped.

# nutpieR 1.7.0

* New `adaptation =` argument on `nutpie_sample()` selects the mass-matrix
  strategy: `"diag"` (default) or `"low_rank"` / `"low-rank"`. Matches Python
  nutpie's API.
* `low_rank_modified_mass_matrix` is soft-deprecated; pass
  `adaptation = "low_rank"` instead.
* New optional `mindepth`, `max_energy_error`, and `extra_doublings`
  arguments expose the matching `nuts-rs` settings.
* `target_accept`, `max_treedepth`, `mass_matrix_gamma`, and
  `mass_matrix_eigval_cutoff` now default to `NULL` and pass through to
  `nuts-rs`'s defaults when unspecified, instead of overwriting them with
  R-side defaults. Same goes for the previously-bumped 800-warmup default
  for low-rank.
* `attr(draws, "sampler_config")` returns the *effective* sampler settings
  as a JSON string. Parse with `jsonlite::fromJSON()` to inspect what the
  sampler actually ran with (including any nuts-rs defaults).
* When `store_mass_matrix = TRUE`, `nutpie_diagnostics(draws)$mass_matrix_inv`
  (and the matching `unconstrained_draw` / `gradient` columns when stored)
  are now numeric matrices with shape `(n_draws * n_chains, ndim_unc)`,
  not lists of vectors. Variable-width list columns still fall back to the
  list representation.
* `mass_matrix_inv` / `mass_matrix_eigvals` / `mass_matrix_stds` rows that
  fall on a non-update draw now carry the most-recent recorded value
  forward instead of surfacing `NA` — the inverse mass matrix is
  piecewise-constant between adapter updates.

# nutpieR 1.6.0

* `nutpie_sample()` validates `seed`: `NA`, negative, fractional, and values
  above `.Machine$integer.max` now error instead of silently producing a
  non-reproducible seed.
* Default `num_warmup` is 800 (was 400) when
  `low_rank_modified_mass_matrix = TRUE`. Explicit values still win.
* `cores = NULL` falls back to 1 when `parallel::detectCores()` returns `NA`.
* Cleaner error output on FFI failures (no more `thread '<unnamed>'
  panicked` noise on stderr). **Error message text has changed for some
  failure paths.**

# nutpieR 1.5.0

* `nutpie_compile_model()` caches compiled artifacts. `stan_file` mode
  drops `<basename>_model.so` next to the `.stan` (cmdstanr-compatible);
  `code` mode hashes into `nutpie_cache_dir()`. Both keys cover BridgeStan
  version and compile flags. Pass `cache = FALSE` (or set
  `NUTPIER_DISABLE_COMPILE_CACHE=1`) to force a recompile;
  `nutpie_clear_cache()` wipes the inline cache.

# nutpieR 1.4.1

* Result conversion no longer copies the draws matrix. The Rust-side buffer
  is already in the right layout for `(n_draws, n_chains, n_params)`, so
  reassigning `dim` in place avoids the full memcpy that `array(.)` was
  doing. Saves ~500 ms per call on a 305 MB result; scales linearly beyond
  there; doubled with `save_warmup = TRUE`.
* `dot_to_bracket()` vectorized — saves ~80 ms on a 1000-parameter model.

# nutpieR 1.4.0

* Memory-efficiency pass on the R/Rust boundary. No API changes for typical
  users. Highlights:
  - New `store_unconstrained` and `store_gradient` flags on `nutpie_sample()`
    control whether the per-draw unconstrained position / gradient columns
    are returned in diagnostics. Default is `FALSE`. Each is an
    `ndim_unc`-wide list-of-vectors — one entry per draw — and on wide
    models they each rival the draws matrix in size. Previously they were
    always materialized into R; opt in only when you need them.
    (Note: nuts-rs 0.17.4 itself ignores these flags and always allocates
    the columns internally; we filter at the R boundary. When nuts-rs
    starts honouring the flags upstream, the gating will move deeper for
    free.)
  - Draws matrix is now written directly into R-allocated memory in Rust,
    eliminating a full intermediate copy during result conversion.
  - `pars` / `include` filtering now happens in Rust before the draws matrix
    is materialized, so memory and copy work scale with the *kept* parameter
    count rather than the full constrained dimension.
  - When `pars` / `include` excludes the entire transformed-parameter and/or
    generated-quantities blocks, bridgestan is told to skip materializing
    those slices per draw. The Stan GQ block (including `*_rng` calls) is
    not run, the per-draw constrained buffer shrinks to the kept slice,
    and the Arrow trace nuts-rs writes is correspondingly smaller. On a
    wide hierarchical model with a meaningful GQ block (`y_rep`,
    `log_lik`), this gives roughly a 38% wall-time reduction and an 80%
    draws-matrix shrinkage when GQ is excluded. Conservative rule: keeping
    any GQ name forces TP to also be materialized, since GQ may reference
    TP.
  - Sampler arguments (`num_draws`, `num_warmup`, `num_chains`, `cores`,
    `max_treedepth`) are now validated for finite, positive values on the R
    side, with a defensive check in Rust before unsigned casts.

# nutpieR 1.3.1

* Bumped `extendr-api` from 0.8.1 to 0.9.0. Fixes the R CMD check NOTE
  about a non-API call to `R_UnboundValue` on R 4.6.0+ (#7) — the symbol
  was removed from R's public API and from extendr in the 0.9.0 release.

# nutpieR 1.3.0

* `nutpie_sample()`'s `init` as a *partial* named list now fills the missing
  parameters independently per chain (using per-chain seeds derived from
  `seed`), instead of broadcasting a single random fill to every chain. Fully
  specified named lists still broadcast one position to all chains.
* Per-chain init resolution no longer evaluates transformed parameters or
  generated quantities when filling missing parameters, so partial inits
  cannot fail on GQ constraint violations during setup. Adds an internal
  `bs_param_constrain_block()` helper that returns block-level constrained
  values only.
* Documented an honest contract for per-chain `init` mapping: when supplying
  list-of-lists or `function(chain_id)` starts, the N positions are guaranteed
  to be distributed one-per-chain, but the mapping from list index /
  `chain_id` to the output chain dimension is not currently guaranteed.
  Threading the true chain id requires an upstream change in nuts-rs.
* `nutpie_sample()`'s `init` argument is now a single entry point that
  dispatches on the input shape:
  - `NULL` (default): per-chain Uniform(-2, 2) draw on the unconstrained
    scale (nuts-rs default).
  - Scalar numeric `x`: per-chain Uniform(-x, x) on the unconstrained scale;
    `init = 0` starts every chain at the origin.
  - Named list: broadcast constrained values (partial inits allowed; missing
    parameters are filled randomly, seeded from `seed`).
  - List of `num_chains` named lists: per-chain constrained inits (each may
    be partial).
  - Function `function(chain_id) ...`: called once per chain, must return a
    (possibly partial) named list of constrained values.
  - Character path(s): one or `num_chains` JSON file paths.
* `init_unconstrained` is removed. Users who need to set unconstrained-scale
  starting points should convert to the constrained scale first (e.g. via
  `nutpie_unconstrain()`'s inverse) and pass those through `init`.
* `init_mean` is soft-deprecated: still works, but emits a warning and
  will be removed in a future version.
* `nutpie_param_names()` gains a `which` argument with values `"block"`
  (default), `"unconstrained"`, and `"full"`. The previous `unconstrained`
  logical argument is soft-deprecated.
* `nutpie_unconstrain()` is reframed as an introspection / debugging helper
  (docstring only — behaviour unchanged).

# nutpieR 1.2.0

* New `init` argument in `nutpie_sample()` accepts initial values on the
  constrained (user-facing) scale, either as a named list (broadcast to all
  chains), a list of `num_chains` named lists (per-chain starts), or a path to
  a CmdStan-style JSON file. Missing parameters are filled with random draws
  per chain (#8).
* New `init_unconstrained` argument takes a named numeric vector (or list of
  vectors) on the unconstrained scale and uses the values exactly — no jitter.
  Names are validated against the model's unconstrained parameter names.
* New introspection helpers: `nutpie_param_names()` returns the parameter
  names (constrained or unconstrained), and `nutpie_unconstrain()` maps a
  constrained named list to the unconstrained vector used internally.
* `init_mean` is preserved for backwards compatibility.
* Internal refactor: a single BridgeStan model is opened per `nutpie_sample()`
  call (via an `ExternalPtr<BSHandle>`) and reused across chain setup and the
  sampler, replacing the previous 5–17 redundant `dlopen` + JSON-parse cycles.
  The constrained → unconstrained mapping now calls BridgeStan's flat
  `param_unconstrain` directly, with no JSON round-trip.

# nutpieR 1.1.0

* New `pars` and `include` arguments in `nutpie_sample()` allow selecting which
  parameters to keep or exclude from the output draws, matching the rstan
  convention. Useful for dropping nuisance parameters to save memory (#5).

# nutpieR 1.0.1

* `init_mean` now accepts a scalar value that is automatically expanded to the
  correct length (#4).
* Draws where generated quantities cannot be computed (e.g. parameter constraint
  violations during transformation) are now filled with `NaN` instead of
  causing a sampler panic, with a warning summarizing the count (#2).
* Warmup draw count is now stored as an attribute on the output (`num_warmup`),
  accessible even when `save_warmup = FALSE` (#3).
* Fixed a flaky test tolerance for the normal model sigma parameter that caused
  Linux build failures on R-universe.

# nutpieR 1.0.0

* Initial release.
* Compile Stan models via BridgeStan (`nutpie_compile_model()`).
* Parallel NUTS sampling in Rust via nuts-rs (`nutpie_sample()`).
* Returns `posterior::draws_array` — compatible with posterior/bayesplot.
* Sampler diagnostics (`nutpie_diagnostics()`), warmup access (`nutpie_warmup_draws()`).
* Also exposed low-rank modified mass matrix adaptation.
* Cross-platform: Windows, macOS, Linux.
