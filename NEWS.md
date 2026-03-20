# nutpieR 1.0.0

* Initial release.
* Compile Stan models via BridgeStan (`nutpie_compile_model()`).
* Parallel NUTS sampling in Rust via nuts-rs (`nutpie_sample()`).
* Returns `posterior::draws_array` — compatible with posterior/bayesplot.
* Sampler diagnostics (`nutpie_diagnostics()`), warmup access (`nutpie_warmup_draws()`).
* Also exposed low-rank modified mass matrix adaptation.
* Cross-platform: Windows, macOS, Linux.
