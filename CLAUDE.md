# nutpieR - R Bindings for the nutpie NUTS Sampler

## What This Is

An R package providing native bindings to the [nuts-rs](https://github.com/pymc-devs/nuts-rs) NUTS sampler (written in Rust), using [BridgeStan](https://github.com/roualdes/bridgestan) as the Stan model backend. The R equivalent of [nutpie](https://github.com/pymc-devs/nutpie) (Python).

The core value proposition: **R users can sample Stan models with nuts-rs without touching Python or reticulate. R is not in the hot loop — parallel chains run entirely in Rust.**

## Architecture

```
R user code
    │
    ▼
R layer (thin functional API)
    │  - compile_model() → calls Rust, which calls bridgestan::compile_model
    │  - nuts_sample() → calls into Rust via extendr
    │  - returns posterior::draws_array (via Arrow transfer)
    │
    ▼
Rust layer (extendr crate)
    │  - compiles Stan models via bridgestan Rust crate (no R BridgeStan package)
    │  - implements nuts-rs Model trait + CpuLogpFunc trait
    │  - CpuLogpFunc::logp calls bridgestan::Model::log_density_gradient
    │  - Sampler::new() runs parallel chains via rayon
    │  - returns draws via Arrow (zero-copy to R via nanoarrow)
    │
    ├── nuts-rs 0.17.x (the NUTS sampler, has "arrow" feature)
    ├── bridgestan 2.7.x (Rust crate, compiles + loads Stan models)
    └── arrow-extendr (Arrow C Data Interface for R ↔ Rust transfer)
```

**Critical architectural point:** During sampling, Rust calls the compiled Stan .so directly through BridgeStan's C ABI. R is NOT in the hot loop. Each chain gets its own `bridgestan::Model` instance (they are Send+Sync via Arc<StanLibrary>). Parallel chains work without R's single-threaded constraint.

**No BridgeStan R package dependency.** The bridgestan Rust crate handles both compilation (`bridgestan::compile_model()` — calls `make` with `STAN_THREADS=true`) and model loading (`bridgestan::open_library()`). It also has a `download-bridgestan-src` feature for fetching the BridgeStan C++ sources from GitHub. This keeps the entire Stan toolchain interaction in Rust.

## Key APIs We're Wrapping

### nuts-rs `CpuLogpFunc` trait (src/cpu_math.rs:640)
```rust
pub trait CpuLogpFunc: HasDims {
    type LogpError: Debug + Send + Sync + Error + LogpError + 'static;
    type FlowParameters;           // () for basic diagonal mass matrix
    type ExpandedVector: Storable<Self>;  // for constrained param transform

    fn dim(&self) -> usize;
    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpError>;
    fn expand_vector<R: Rng>(&mut self, rng: &mut R, array: &[f64]) -> Result<Self::ExpandedVector, CpuMathError>;
}
```

### nuts-rs `Model` trait (factory for per-chain instances)
```rust
pub trait Model: Send + Sync + 'static {
    type Math<'model>: Math where Self: 'model;
    fn math<R: Rng>(&self, rng: &mut R) -> Result<Self::Math<'_>>;
    fn init_position<R: Rng>(&self, rng: &mut R, position: &mut [f64]) -> Result<()>;
}
```

### bridgestan Rust crate
```rust
let lib = bridgestan::open_library(path)?;           // Arc-able, Send+Sync
let model = bridgestan::Model::new(Arc::clone(&lib), data, seed)?;  // Send+Sync
model.log_density_gradient(theta_unc, propto, jacobian, grad)?;      // hot path
model.param_constrain(theta_unc, include_tp, include_gq, out, rng)?; // for expand_vector
model.param_unc_num();  // dimension
model.param_names(true, true);  // parameter names
```

### nuts-rs `Sampler` (parallel chain orchestration)
```rust
Sampler::new(model, settings, trace_config, num_cores, callback)?;
// wait_timeout takes ownership (self, not &mut self)
// Returns SamplerWaitResult enum, NOT Result
match sampler.wait_timeout(duration) {
    SamplerWaitResult::Trace(traces) => { /* Vec<ArrowTrace> */ },
    SamplerWaitResult::Timeout(sampler) => { /* get sampler back */ },
    SamplerWaitResult::Err(e, partial) => { /* anyhow::Error + Option<traces> */ },
}
```

## How nutpie Does It (our reference)

Key file: `nutpie/src/stan.rs`
- `StanModel` wraps `Arc<bridgestan::Model<Arc<StanLibrary>>>` + parameter metadata
- `StanDensity<'model>` implements `CpuLogpFunc`:
  - `logp()` → `model.inner.log_density_gradient(position, true, true, grad)`
  - `expand_vector()` → `model.inner.param_constrain()` + Fortran→C order transpose
  - All errors marked recoverable (becomes divergence, not panic)
- `StanModel` implements `Model` trait (factory):
  - `math()` creates new `CpuMath<StanDensity>` per chain
  - `init_position()` fills with standard normal draws
- Uses `Sampler::new()` with `ArrowConfig` for trace storage + rayon thread pool

## Project Structure (Target)

```
nutpieR/
├── DESCRIPTION
├── NAMESPACE
├── R/
│   ├── nutpieR-package.R   # package-level doc
│   ├── compile.R            # compile_model() — thin wrapper calling Rust
│   ├── sample.R             # nuts_sample() — calls into Rust via extendr
│   ├── results.R            # convert Arrow output to posterior::draws_array
│   └── extendr-wrappers.R   # auto-generated by rextendr
├── src/
│   ├── Makevars{,.win}
│   ├── entrypoint.c
│   └── rust/
│       ├── Cargo.toml        # nuts-rs, extendr-api, bridgestan, arrow-extendr
│       └── src/
│           ├── lib.rs         # extendr module definition
│           ├── model.rs       # CpuLogpFunc + Model trait impl using BridgeStan
│           └── sampler.rs     # sampling config, execution, results collection
├── tests/testthat/
└── ai_context/               # reference repos (build-ignored)
```

## Building & Testing

```r
devtools::load_all()    # Load package (compiles Rust)
devtools::test()        # Run tests
devtools::check()       # Full R CMD check
```

## Windows Environment Notes

This project is developed on Windows with Git Bash (MINGW64).

### Rscript newline handling
Multi-line `Rscript -e` commands cause segfaults due to newline handling.

**Workarounds:**
- Use semicolons: `Rscript -e "x <- 1; y <- 2; print(x+y)"`
- Use multiple -e flags: `Rscript -e "x <- 1" -e "print(x)"`
- Write to a temp .R file for complex scripts

### Spaces in paths (CRITICAL)
The project lives under `Documents/R packages/nutpieR` — the space in `R packages` breaks:
- **dlltool / assembler**: Rust's import library generation splits `--temp-prefix` at spaces. The `Makevars.win.in` template redirects `TARGET_DIR` and `CARGO_HOME` to `$(TEMP)/nutpieR-*` (no spaces) to work around this.
- **CARGO_HOME export**: Must be quoted in Makevars (`export CARGO_HOME="$(CARGOTMP)"`).
- **Any new Makevars paths** that touch `$(CURDIR)` will hit this — always use the temp dir pattern or quote paths.

### extendr `Result` type conflict
`extendr_api::prelude::*` imports `Result<T>` (1 generic param), which shadows `std::result::Result<T, E>`. In trait impls for nuts-rs, use fully qualified `std::result::Result<T, E>`.

## Spike Learnings (Validated)

### What works
- **extendr + nuts-rs on Windows**: Full dependency tree (nuts-rs 0.17.4, faer 0.24, rayon, pulp, arrow 57) compiles cleanly with `x86_64-pc-windows-gnu` target and Rust 1.94.0.
- **Parallel sampling via rayon**: Multiple chains run in parallel without issues (rayon threads don't call R API).
- **Arrow trace output**: `ArrowConfig` produces `Vec<ArrowTrace>` (one per chain), each with `posterior` and `sample_stats` RecordBatches.

### Arrow output format details
- The `posterior` RecordBatch has a `value` column of type `LargeList(Float64)` (NOT `FixedSizeList`). Column metadata has `dims` and `shape` keys.
- The RecordBatch includes **both warmup and post-warmup draws** (`num_tune + num_draws` rows). Must skip the first `num_tune` rows when extracting draws.
- `Vec<f64>` implements `Storable<P>` out of the box via nuts-storable — produces a single `value` column.

### rand 0.10 API
- `random_range()` requires `use rand::RngExt` (it's an extension trait in rand 0.10, not on `Rng` directly).

## R Dependencies

| Package | Role |
|---------|------|
| `posterior` | draws_array/draws_df format for MCMC output |
| `nanoarrow` | Arrow C Data Interface for receiving draws from Rust |

No dependency on the BridgeStan R package — compilation and model loading are handled entirely in Rust.

## Rust Dependencies

| Crate | Role |
|-------|------|
| `nuts-rs` (~0.17, features: arrow) | The NUTS sampler + Arrow trace storage |
| `extendr-api` (0.8.1) | R ↔ Rust FFI |
| `bridgestan` (~2.7, features: download-bridgestan-src) | Compiles + loads Stan models |
| `arrow-extendr` | Arrow transfer R ↔ Rust via nanoarrow |
| `arrow` (~57, default-features = false) | Arrow arrays (must match nuts-rs version) |
| `anyhow` | Error handling (required by Model trait signatures) |
| `thiserror` | Error type derivation |
| `rand` (~0.10) | RNG (must match nuts-rs version) |
| `rayon` | Parallel chains (pulled in by nuts-rs) |

## Design Decisions

- **extendr** for R-Rust bridge (not raw `.Call` / `.C`)
- **No BridgeStan R package** — Rust handles compilation (`bridgestan::compile_model`) and loading
- **Arrow** for data transfer from the start (nuts-rs has `arrow` feature, use `arrow-extendr` + `nanoarrow`)
- **DiagGradNutsSettings** for MVP (standard diagonal mass matrix NUTS)
- **`posterior::draws_array`** as the output format (chains × iterations × parameters)
- **Functional R API** (not R6 — simpler, matches Stan ecosystem conventions)
- **Stan-only** for MVP (no R-function-as-logp backend)
- **Not targeting CRAN** — no vendoring or MSRV constraints. Windows support matters.

## Remaining Spike Work

- [ ] **BridgeStan integration** (Phase 1, step 4-5): Add bridgestan crate, implement CpuLogpFunc calling BridgeStan, test with a precompiled Stan model. Requires libclang/LLVM for bindgen on Windows.

## Phase 2: MVP

6. Model compilation from Rust (`bridgestan::compile_model` + `download_bridgestan_src`)
7. Parallel chains via `Sampler::new()` with `ArrowConfig` trace storage
8. Arrow transfer to R via `arrow-extendr` + `nanoarrow`, wrap as `posterior::draws_array`
9. Settings: warmup, draws, chains, seed, max_treedepth
10. Basic error handling and user messages

## Phase 3: Polish

11. Progress reporting during sampling
12. Sampler diagnostics (divergences, treedepth, step size, energy)
13. `expand_vector` support (constrained parameter transforms via `param_constrain`)
14. Documentation and vignettes

## Known Risks & Uncertainties

1. ~~**extendr + nuts-rs compilation on Windows**~~: ✅ VALIDATED — builds cleanly
2. **bridgestan on Windows**: `libloading` + `.dll` loading may need special path handling
3. **bridgestan build.rs uses bindgen**: requires `libclang` / `LIBCLANG_PATH` on Windows
4. **Arrow version alignment**: nuts-rs uses arrow 57.0, arrow-extendr must support same major version
5. ~~**nuts-rs edition 2024**~~: ✅ Rust 1.94.0 handles this fine
6. **STAN_THREADS=true**: bridgestan Rust `compile_model` already sets this env var (confirmed in source), but pre-compiled models must also have it

## Reference Repos (in ai_context/)

- `ai_context/nutpie/` — THE reference implementation (Python wrapper for nuts-rs)
- `ai_context/nuts-rs/` — The NUTS sampler in Rust
- `ai_context/bridgestan/` — BridgeStan C/Rust/R/Python bindings

Key files to study:
- `nutpie/src/stan.rs` — BridgeStan integration, CpuLogpFunc impl
- `nutpie/src/wrapper.rs` — Sampler orchestration, settings, results
- `nuts-rs/src/cpu_math.rs` — CpuLogpFunc trait definition
- `nuts-rs/src/sampler.rs` — DiagGradNutsSettings, Sampler, parallel chains
- `nuts-rs/src/model.rs` — Model trait
- `bridgestan/rust/src/bs_safe.rs` — Rust API for loading/querying Stan models
