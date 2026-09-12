# Native kernel producer guide

BYOK (bring your own kernel) lets a trusted native library provide a Stan
model's log density and gradient. The library runs in the sampler's native
hot loop; BridgeStan remains the reference model and owns initialization,
transforms, names, transformed parameters, generated quantities, and output.

This is the maintained producer guide. The header
`inst/include/nutpier_kernel_v1.h` (installed as `include/nutpier_kernel_v1.h`)
is the source of truth for the ABI, ownership, status codes, buffers, and
thread safety. Read it before writing a kernel. Function help covers the R API;
this guide covers the workflow.

For a runnable end-to-end example, start with `run.R`; it creates a temporary
working directory, copies `gaussian.c` and `gaussian.stan`, builds the library,
and runs the checks and a sample. The snippets below are illustrative and
assume a writable directory containing those files.

Compile a Stan reference as usual, then attach a native kernel and the data
for one immutable binding. The kernel replaces only density and gradient
evaluation; R is not in that loop. A binding is session-local, so save the
reference and source library, not the binding itself, and reattach after
restoring an R session.

## Workflow

1. Compile the reference and choose the data.
2. Ask the reference for its layout with `nutpie_kernel_layout()`.
3. Build the native library against `nutpier_kernel_v1.h`.
4. Attach the library with the same data.
5. Run the default random check, then the reference check.
6. Run a short smoke sample before a real run.

The helper is authoritative for the ABI layout:

```r
library(nutpieR)
example_dir <- system.file("examples/byok", package = "nutpieR",
                          mustWork = TRUE)
reference <- nutpie_compile_model(file.path(example_dir, "gaussian.stan"))
data <- list(n = 2L, mu = 1, sigma = 2)
layout <- nutpie_kernel_layout(reference, data)
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
bound <- nutpie_attach_kernel(reference, library_path, data = data)

random <- nutpie_validate_kernel(bound, seed = 42)
print(random)
reference_check <- nutpie_validate_kernel(bound, seed = 42, method = "reference")
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
`nutpie_attach_kernel(reference, library_path, data = new_data)` again and check
the new binding.

## Kernel contract and checks

At evaluation time, status `0` means success. Status `1` means an expected
domain rejection or nonfinite proposal. Status `2` means a fatal internal or
setup failure; other statuses are treated as fatal. A status-1 evaluation can
be rejected by the sampler. A fatal error stops the run. Keep the native code
trusted: it runs inside R and can crash or corrupt the process.

`nutpie_validate_kernel()` is an advisory numerical correctness comparison. It
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

## Example files

`gaussian.c` is a small data-bound producer. Its parser is deliberately limited
to the example's flat numeric `n`, `mu`, and `sigma`; use a maintained JSON
parser for production code. `run.R` is the runnable end-to-end workflow:

```r
source(system.file("examples/byok/run.R", package = "nutpieR"))
```

Start with the installed header and the layout returned by
`nutpie_kernel_layout()`.
