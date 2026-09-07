# Experimental BYOK example

Run from an installed nutpieR:

```r
example <- source(system.file("examples/byok/run.R", package = "nutpieR"))$value
```

The script compiles `gaussian.c` with `R CMD SHLIB`, compiles its Stan reference,
binds runtime data, explicitly checks numerical agreement, then samples. Build
files stay in a session temporary directory. A C/C++ toolchain is required.

The producer implements `nutpier_kernel_v1.h`. It owns immutable Gaussian data
and needs no workspace allocation. It checks dimension and ordered names at
bind time. The small parser accepts flat numeric `n`, `mu`, and `sigma`, including
reordered keys; it is illustrative, not a general JSON parser. Use a maintained
parser in a real producer. This code does not generate or compile other kernels.

The density follows BridgeStan `propto=true, jacobian=true`. Sampling is allowed
without checking, but finite-point agreement is useful evidence. It is not a
native memory-safety certificate. Native code runs inside R and must satisfy
the header's ownership, ABI and thread-safety contract.
