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
