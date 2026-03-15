# nutpieR

R bindings for the [nuts-rs](https://github.com/pymc-devs/nuts-rs) NUTS sampler, using [BridgeStan](https://github.com/roualdes/bridgestan) for Stan model evaluation. The R equivalent of [nutpie](https://github.com/pymc-devs/nutpie) (Python).

R is not in the hot loop -- parallel chains run entirely in Rust.

## Installation

<!-- TODO: uncomment when R-universe is set up
```r
install.packages("nutpieR", repos = "https://andytimm.r-universe.dev")
```
-->

Install from source (requires a [Rust toolchain](https://rustup.rs/)):

```r
remotes::install_github("andytimm/nutpieR")
```

### System requirements

- **Rust**: `rustc >= 1.85` and `cargo` ([install via rustup](https://rustup.rs/))
- **C++ toolchain** (for compiling Stan models):
  - Windows: [Rtools](https://cran.r-project.org/bin/windows/Rtools/)
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Linux: `build-essential` (Debian/Ubuntu) or equivalent

## Usage

```r
library(nutpieR)

# Compile a Stan model
model <- nutpie_compile_model(code = "
  data { int<lower=0> N; array[N] int<lower=0,upper=1> y; }
  parameters { real<lower=0,upper=1> theta; }
  model { theta ~ beta(1, 1); y ~ bernoulli(theta); }
")

# Sample
draws <- nutpie_sample(
  model,
  data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
  num_draws = 1000,
  num_chains = 4,
  seed = 42
)

# Returns a posterior::draws_array -- works with all posterior/bayesplot tools
posterior::summarize_draws(draws)
```

You can also compile from a `.stan` file:

```r
model <- nutpie_compile_model(stan_file = "my_model.stan")
```

### Diagnostics

```r
# Sampler diagnostics (divergences, treedepth, energy, etc.)
nutpie_diagnostics(draws)

# Access warmup draws (if save_warmup = TRUE)
draws <- nutpie_sample(model, data = dat, save_warmup = TRUE)
nutpie_warmup_draws(draws)
nutpie_warmup_diagnostics(draws)
```

### Sampling parameters

```r
draws <- nutpie_sample(
  model,
  data = dat,
  num_draws = 1000,       # post-warmup draws per chain
  num_warmup = 1000,      # warmup draws per chain
  num_chains = 4,         # number of chains
  cores = 4,              # parallel cores
  seed = 604,              # RNG seed
  max_treedepth = 10,     # maximum tree depth
  target_accept = 0.8,    # target acceptance rate
  refresh = 100,          # progress every N draws (0 = off)
  save_warmup = FALSE,    # save warmup draws?
  store_divergences = FALSE,  # store divergence details?
  store_mass_matrix = FALSE   # store mass matrix?
)
```

### Low-rank mass matrix adaptation

For models with correlated parameters, nutpieR supports low-rank modified mass
matrix adaptation from nuts-rs. This captures posterior correlations more
effectively than the default diagonal mass matrix, and can significantly improve
sampling efficiency on challenging geometries.

```r
draws <- nutpie_sample(
  model,
  data = dat,
  low_rank_modified_mass_matrix = TRUE,  # enable low-rank adaptation
  mass_matrix_gamma = 1e-5,              # regularisation (default)
  mass_matrix_eigval_cutoff = 2.0        # eigenvalue cutoff (default)
)
```

When enabled, the sampler defaults to 800 warmup draws (vs 400 for diagonal) to
allow the low-rank structure to stabilize.

## How it works

nutpieR compiles Stan models via the BridgeStan Rust crate and samples using the nuts-rs NUTS sampler. During sampling, Rust calls the compiled Stan shared library directly through BridgeStan's C ABI -- R is not involved in the sampling loop. Each chain runs on its own thread via rayon.

Results are transferred from Rust to R via Apache Arrow (zero-copy) and returned as a `posterior::draws_array`.

## License

MIT
