# nutpieR - R Bindings for the nutpie NUTS Sampler

## What This Is

An R package providing native bindings to the [nuts-rs](https://github.com/pymc-devs/nuts-rs) NUTS sampler (written in Rust), using [BridgeStan](https://github.com/roualdes/bridgestan) as the Stan model backend. The R equivalent of [nutpie](https://github.com/pymc-devs/nutpie) (Python).

**R users can sample Stan models with nuts-rs without touching Python or reticulate. R is not in the hot loop — parallel chains run entirely in Rust.** Benchmarked at ~4x faster than CmdStanR on a 1040-parameter hierarchical model.

## Current State

The package is **functional** — compile Stan models, sample with parallel chains, get `posterior::draws_array` output with diagnostics. All core MVP features are implemented.

### What works
- `nutpie_compile_model(stan_file)` or `nutpie_compile_model(code = "...")` — compiles Stan models via BridgeStan
- `nutpie_sample(model, data, ...)` — parallel NUTS sampling with progress bars
- Sampling parameters include: `num_draws`, `num_warmup`, `num_chains`, `cores`, `seed`, `max_treedepth`, `target_accept`, `refresh`, `verbose`
- Single shape-dispatched `init` entry point: `NULL` | scalar (Uniform(-x, x)) | named list (constrained, partial OK) | list of `num_chains` lists | `function(chain_id)` | JSON path(s). `init_mean` is soft-deprecated.
- Returns `posterior::draws_array` — works with all posterior/bayesplot tooling
- Diagnostics: `nutpie_diagnostics()` returns divergences, treedepth, energy, step size, etc.
- Warmup access: `nutpie_warmup_draws()` and `nutpie_warmup_diagnostics()` when `save_warmup = TRUE`
- Optional `store_divergences` and `store_mass_matrix` for detailed debugging
- Low-rank modified mass matrix adaptation (`low_rank_modified_mass_matrix = TRUE`)
- Cross-platform: builds on Windows, macOS, and Linux without LLVM/libclang

### What's next
- Vignettes
- Upstream bridgestan allowlist PR

## Architecture and source map

R validates inputs and calls Rust through extendr. Native Rust worker threads
run the chains; R is not in the log-density hot loop. Chains share an immutable
BridgeStan model and own density/RNG state. Optional density kernels also get
per-chain workspaces.

nuts-rs stores traces in Arrow. Rust copies selected values into R-owned
matrices, which R reshapes into `posterior::draws_array`; there is no
arrow-extendr/nanoarrow transfer.

- `R/compile.R`, `R/cache.R`: compilation and persistent model cache.
- `R/sample.R`, `R/init.R`: sampling orchestration and initialization.
- `R/results.R`, `R/diagnostics.R`: matrix reshaping and diagnostics accessors.
- `R/progress.R`: cli/text progress and sampling summaries.
- `R/density-kernel*.R`, `R/walnuts.R`: optional kernel and WALNUTS interfaces.
- `R/extendr-wrappers.R`: generated extendr wrappers.
- `src/rust/src/lib.rs`: extendr boundaries, settings, and result materialization.
- `src/rust/src/sampling.rs`: native sampler execution and progress snapshots.
- `src/rust/src/model.rs`: shared BridgeStan model and per-chain density state.
- `src/rust/src/private_tbb.rs`: private macOS TBB bundles.
- `src/rust/src/density_kernel.rs`, `src/rust/src/walnuts.rs`: optional backends.
- `tests/testthat/`: unit and integration tests; models live in `test_models/`.

## Building & Testing

```sh
# Install (release Rust build unless DEBUG is set)
NOT_CRAN=TRUE Rscript -e 'devtools::install(quick = TRUE)'

# Default tests; a warm compile cache avoids recompiling Stan fixtures
Rscript -e 'devtools::test()'

# Slow posteriordb reference comparisons
NUTPIER_RUN_SLOW_TESTS=1 Rscript -e 'devtools::test()'

# Full check
Rscript -e 'devtools::check()'
```

See `tests/README.md` for the full list of optional test env vars and the cross-implementation comparison script.

For IDE-specific progress bugs, Playwright can drive the actual R Console (not
the Terminal tab) in an isolated [RStudio](https://github.com/rstudio/rstudio/blob/main/e2e/rstudio/README.md) or [Positron](https://github.com/posit-dev/positron/blob/main/test/e2e/README.md) instance. This is an optional debugging tool, not a CI gate. Leave color and Unicode at their defaults.
If warnings look monochrome, check `Sys.getenv("NO_COLOR")` and
`cli::num_ansi_colors()` in the IDE’s R Console. Agent-launched IDEs can inherit
`NO_COLOR=1`; for a color preview, unset it in the demo launcher only rather
than forcing color in nutpieR or changing IDE preferences.

**Testing notes:** `devtools::test()` (which uses `load_all()`) is the normal
loop; `devtools::test(filter = "progress")` iterates on progress tests.

- Rust build mode is controlled by `DEBUG`: unset uses `--release`; any
  nonempty value uses Cargo's default debug profile. Debug builds retain the
  target directory. Release builds delete it after linking: `src/rust/target`
  on Unix, `$(TEMP)/nutpieR-rust-target` on Windows. Release installs can take
  minutes; use `cargo check` in `src/rust/` for fast Rust-only feedback.
- An empty extendr `Strings` has triggered a debug-build
  `slice::from_raw_parts` panic in `test-compile-cache.R`. If encountered,
  check the actual Rust build mode rather than assuming `devtools::test()`
  selects debug. To validate the installed release package, unset `DEBUG`, run
  the install command above, then run
  `Rscript -e 'library(nutpieR); library(testthat); test_dir("tests/testthat")'`.
  Under `test_dir`, mocked base `interactive()` may not take effect, causing
  spurious failures/skips in auto-progress mode tests; use `devtools::test()`
  for those tests.

## System Requirements

- **Rust toolchain**: `rustc >= 1.88`, `cargo` (edition 2021)
- **C++ toolchain**: needed to compile Stan models with BridgeStan’s Makefile
  - Windows: rtools45
  - macOS: Xcode Command Line Tools
  - Linux: build-essential (gcc, make)
- **No LLVM/libclang needed** — bridgestan bindings are pre-generated

## BridgeStan Patch

We carry a patched bridgestan 2.8.0 in `src/rust/patches/bridgestan-2.8.0/`, referenced via `[patch.crates-io]` in Cargo.toml. Three changes from upstream:

1. **Pre-generated bindings** (`src/pregenerated_bindings.rs`): bindgen output for 22 `bs_*` FFI symbols, generated once. The `build.rs` copies this file instead of running bindgen. Eliminates LLVM/libclang requirement on all platforms.

2. **Root cause**: upstream bridgestan runs bindgen with `dynamic_link_require_all(true)` and no allowlist. On Windows, system headers inject CRT symbols that `from_library()` then tries to resolve from the Stan model DLL — causing `GetProcAddress` error 127. Our pre-generated bindings contain only BridgeStan symbols; the patched `build.rs` copies them and does not run bindgen or apply an allowlist. An upstream `bs_.*` allowlist would address the original cause.

3. **OS trust store for downloads** (`src/download.rs`): `download_bridgestan_src()` builds a ureq agent with `RootCerts::PlatformVerifier` (ureq `platform-verifier` feature) instead of the bundled webpki-roots, so TLS-intercepting corporate proxies with a system-installed CA work (issue #30). Also worth upstreaming.

We also pin the vendored crate to `edition = "2021"` (upstream 2.8.0 ships edition 2024). The pre-generated bindings are a bindgen edition-2021 artifact, so under edition 2024 they trip 43 `unsafe_op_in_unsafe_fn` warnings; staying on 2021 keeps the build warning-free without suppression. `bridgestan.h` was byte-identical from 2.7.0 → 2.8.0, so `pregenerated_bindings.rs` carried over unchanged — no bindgen/libclang regeneration was needed for this bump.

**Regenerating bindings**: If BridgeStan updates its C API, use a temporary
bindgen-enabled build with both the upstream bindgen build dependency and its
binding-generation `build.rs`, restricted to `bs_.*` symbols. Generate
edition-2021 bindings, inspect the exported symbols, and copy the resulting
`bindings.rs` to `src/pregenerated_bindings.rs`. Adding only the dependency is
not enough: our patched `build.rs` merely copies the existing file. Restore the
copy-only build afterward so users still need no libclang.

## Windows Environment Notes

### Rscript newline handling
Multi-line `Rscript -e` commands cause segfaults. Use semicolons or multiple `-e` flags.

### Spaces in paths
The `Makevars.win.in` redirects `TARGET_DIR` and `CARGO_HOME` to `$(TEMP)/nutpieR-*` (no spaces) to avoid dlltool/assembler issues. `nutpie_compile_model()` copies stan files to temp dirs if paths contain spaces (make can't handle them).

### extendr `Result` type conflict
`extendr_api::prelude::*` imports `Result<T>` (1 generic param), shadowing `std::result::Result<T, E>`. Use fully qualified `std::result::Result<T, E>` in nuts-rs trait impls.

### TBB DLL
Stan models with `STAN_THREADS=true` need `tbb.dll` on PATH. `add_tbb_to_path()` in `model.rs` handles this by scanning `~/.bridgestan/` for TBB directories.

## Dependencies and upstream APIs

Use `DESCRIPTION` for R dependencies and `src/rust/Cargo.toml` / `Cargo.lock`
for Rust versions. Core dependencies include nuts-rs 0.19.1 with Arrow,
extendr-api 0.9.0, patched BridgeStan 2.8, and Arrow 59. Keep Arrow and rand
compatible with nuts-rs. The diagonal settings type is `DiagNutsSettings`;
low-rank adaptation uses `LowRankNutsSettings`.

For current trait signatures, start with the working implementations in
`src/rust/src/model.rs` and the resolved nuts-rs source. In particular,
`Model::math` receives `Arc<Self>`, and `init_position` receives a chain ID;
do not infer chain identity from worker scheduling.

Reference checkouts may be available under build-ignored `ai_context/`
(`nutpie`, `nuts-rs`, `bridgestan`). Check their versions against the resolved
Cargo dependencies before using them as API references.

## Design Decisions

- extendr for the R–Rust boundary; native Rust chain execution.
- R runs BridgeStan’s Makefile so Stop can clean up the compiler process tree. Rust prepares BridgeStan and loads the compiled library.
- Arrow traces copied into R-owned matrices; `posterior::draws_array` output.
- Functional R API, not R6, matching Stan ecosystem conventions.
- R-universe binary distribution; not targeting CRAN. Rust MSRV is declared
  in `DESCRIPTION` and `Cargo.toml` and checked by `tools/msrv.R`.
