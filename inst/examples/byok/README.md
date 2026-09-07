# Experimental BYOK example

Run from an installed nutpieR:

```r
example <- source(system.file("examples/byok/run.R", package = "nutpieR"))$value
```

You need a C/C++ toolchain. The script compiles `gaussian.c` with `R CMD SHLIB`
and its Stan reference, binds data, checks pilot and broad random points separately, then samples.
Build files stay in a session temporary directory.

## Writing a kernel

The C library implements `nutpier_kernel_v1.h`. It copies Gaussian data at bind
time, keeps it immutable, and needs no workspace allocation. It checks the
parameter dimension and ordered names before accepting a binding.

The example parser accepts flat numeric `n`, `mu`, and `sigma`, including
reordered keys. It is not a general JSON parser; use a maintained parser for
production code. This example does not generate or compile other kernels.

The density follows BridgeStan `propto=true, jacobian=true`. Sampling does not
require validation. Agreement at tested points is evidence, not a guarantee of
memory safety. Native code runs inside R and must satisfy the header's
ownership, ABI and thread-safety contract.
