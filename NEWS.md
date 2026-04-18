# nutpieR 1.3.0

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
