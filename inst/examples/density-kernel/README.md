# Writing a density kernel

Bring Your Own Density Kernel lets you attach a custom density kernel. A density
kernel is a shared library that computes a Stan model's unconstrained log density
and full gradient. It replaces BridgeStan density evaluation, and the NUTS sampler
is unchanged. BridgeStan remains the
reference and handles initialization, transforms, names, transformed
parameters, generated quantities, and output.

This guide covers how to write and check a kernel. The header
`inst/include/nutpier_density_kernel_v1.h` (installed as `include/nutpier_density_kernel_v1.h`)
is the source of truth for the ABI, ownership, status codes, buffers, and
thread safety. Read it before writing a kernel. Function help covers the R API;
this guide covers the workflow.

For a runnable end-to-end example, start with `run.R`; it creates a temporary
working directory, copies `gaussian.c` and `gaussian.stan`, builds the library,
and runs the checks and a sample. The snippets below are illustrative and
assume a writable directory containing those files.

Compile a Stan reference, then attach the kernel with one data snapshot. R
is not in the evaluation loop. The resulting binding works only in the current
session, so save the reference and shared library, not the binding, and reattach
after restoring an R session.

## Workflow

1. Compile the reference and choose the data.
2. Ask the reference for its layout with `nutpie_density_layout()`.
3. Build the native library against `nutpier_density_kernel_v1.h`.
4. Attach the library with the same data.
5. Run the default random check, then the reference check.
6. Run a short smoke sample before a real run.

The helper is authoritative for the ABI layout:

```r
library(nutpieR)
example_dir <- system.file("examples/density-kernel", package = "nutpieR",
                          mustWork = TRUE)
reference <- nutpie_compile_model(file.path(example_dir, "gaussian.stan"))
data <- list(n = 2L, mu = 1, sigma = 2)
layout <- nutpie_density_layout(reference, data)
layout$names
layout$layout
```

`names` contains BridgeStan's exact ordered raw names. `layout` joins those
names with newlines and has no trailing newline. For this example the value is
`"x.1\nx.2"`. R output may display the same coordinates as `x[1]` and
`x[2]`; that bracket notation is not the native ABI. Do not infer transformed
coordinate order from declarations alone. Use the helper with the exact data
you will attach.

Build from the directory containing `gaussian.c`:

```r
include <- system.file("include", package = "nutpieR", mustWork = TRUE)
Sys.setenv(PKG_CPPFLAGS = paste0('-I"', include, '"'))
system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", "gaussian.c"))
```

Use the library that the build actually produced. `R CMD SHLIB` follows
`.Platform$dynlib.ext`; macOS Cargo builds commonly produce `.dylib` instead.
Do not guess the suffix or move between build directories. Run the build from
the intended directory, then use `normalizePath()` on the resulting file:

```r
library_path <- normalizePath(
  file.path(getwd(), paste0("gaussian", .Platform$dynlib.ext)),
  mustWork = TRUE
)
bound <- nutpie_attach_density_kernel(reference, library_path, data = data)

random <- nutpie_validate_density_kernel(bound, seed = 42)
print(random)
reference_check <- nutpie_validate_density_kernel(bound, seed = 42, method = "reference")
print(reference_check)
stopifnot(random$status == "pass", reference_check$status == "pass")

draws <- nutpie_sample(bound, num_draws = 100, num_warmup = 100,
                       num_chains = 2, seed = 42, progress = "none")
```

The default check uses broad random unconstrained points. `method = "reference"`
first obtains a short BridgeStan-only pilot (200 warmup iterations and the
requested retained points), then adds four broad random points. The pilot's
elapsed time is metadata in `$pilot`; it is not a benchmark. Pass explicit
`points` when you need to replay a disagreement or debug one coordinate.

The data are snapshotted when you attach. A bound model uses that snapshot, so
`nutpie_sample(bound, data = ...)` is an error. To change data, call
`nutpie_attach_density_kernel(reference, library_path, data = new_data)` again and check
the new binding.

## Density kernel contract and checks

At evaluation time, status `0` means success. Status `1` means an expected
domain rejection or nonfinite proposal. Status `2` means a fatal internal or
setup failure; other statuses are treated as fatal. A status-1 evaluation can
be rejected by the sampler. A fatal error stops the run. Keep the native code
trusted: it runs inside R and can crash or corrupt the process.

`nutpie_validate_density_kernel()` is an advisory numerical correctness comparison. It
is not a benchmark and cannot certify memory safety, thread safety, or
agreement away from the points tested. Do not loosen tolerances to hide a
disagreement. Explicit points are useful for replay, not coverage claims.

The kernel must match BridgeStan's `propto = TRUE, jacobian = TRUE` target in
the reference's unconstrained coordinates. Sampling statements may drop terms
that are constant in the parameters under `propto = TRUE`; explicit
`target += *_lpdf` and `*_lpmf` calls retain their constants in the BridgeStan
reference. Match the form used by the model. Include transform Jacobians and
their derivatives, and keep the algebra stable. A possible constant offset is a
diagnosis of a translation problem, never a calibration target.

For a lower-bounded scalar, `x = L + exp(q)`: add `q` to the log target, with
chain-rule gradient `dlogp/dx * exp(q) + 1`. For a doubly bounded scalar, let
`p = inv_logit(q)` and `x = L + (U - L) * p`: add
`log(U - L) + log(p) + log1p(-p)`, with gradient
`dlogp/dx * (U - L) * p * (1 - p) + 1 - 2 * p`. Use stable log-sigmoid forms.
This reminder is not exhaustive; query BridgeStan and match it for structured
constraints.

## Example files

`gaussian.c` is a small data-bound kernel. Its parser is deliberately limited
to the example's flat numeric `n`, `mu`, and `sigma`; use a maintained JSON
parser for production code. `run.R` is the runnable end-to-end workflow:

```r
source(system.file("examples/density-kernel/run.R", package = "nutpieR"))
```

Start with the installed header and the layout returned by
`nutpie_density_layout()`.
