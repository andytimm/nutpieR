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
