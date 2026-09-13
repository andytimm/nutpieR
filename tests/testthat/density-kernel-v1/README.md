# Density kernel v1 test producers

These C fixtures test the installed kernel ABI. Run native integration tests
with a release installation and `NUTPIER_RUN_DENSITY_KERNEL_TESTS=1`. Pure checker tests
do not load these libraries.

`gaussian.c` accepts flat numeric `n`, `mu`, `sigma` data. Key order, whitespace
and equivalent number spellings do not matter. It ignores extra numeric fields
but rejects nested or string-valued fields. This is a test parser, not a general
JSON implementation. The reference is `gaussian.stan`, using `propto=true` and
`jacobian=true`.

Compile a fresh copy in a temporary directory with `R CMD SHLIB gaussian.c`.
Set `PKG_CPPFLAGS` to `-I<installed nutpieR/include> -DFIXTURE_MODE=<mode>`.
Never rebuild a library while a process still holds it open.

Modes: 0 runtime data, 1 fixed n=2/mu=1/sigma=2, 2 NULL workspace,
3 constant offset, 4 swapped gradient, 5 untouched final gradient (host must
poison outputs), 6 successful NaN, 7 domain status, 8 fatal poisoned outputs,
9 unknown status, 10 full-capacity unterminated error, 11 workspace failure,
12 bind failure after allocation, 13 wrong ABI version, 14 history dependence.
Normal and fixed variants share one lifecycle ABI and check ordered `x.1`,
`x.2`, ... unconstrained names. Changing `n` changes the parameter dimension.

Run returned-error and sampling failure cases only in a child R process with a
time limit. These fixtures do not intentionally crash, hang, overflow buffers,
or send a foreign exception across the ABI. Passing these tests does not show
that the host can recover from those unsafe cases.

Modes 15 and 16 use `positive.stan`: positive vector coordinates transform as
`x = exp(q)`. Mode 15 supplies `sum(-0.5 * ((exp(q)-mu)/sigma)^2 + q)` and
its full gradient. Mode 16 deliberately omits the Jacobian and must fail.
The reference includes transformed parameters and RNG generated quantities
for separate sampling/output regression tests. These checker cases do not
assume successful validation proves generated-quantity or sampling behavior.
