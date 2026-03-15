# Windows vs macOS/Linux Build Notes

## The Core Problem: bindgen + Windows System Headers

bridgestan's `build.rs` uses bindgen to parse `bridgestan.h`, which `#include`s `<stddef.h>` and `<stdbool.h>`. On macOS/Linux these are clean type-only headers. On Windows, they pull in MSVC CRT declarations — functions like `_errno`, `__security_init_cookie`, `__va_start`, `__threadid`, etc.

With bridgestan's `dynamic_link_require_all(true)` setting and **no allowlist filters**, bindgen generates a `from_library()` function that tries to `GetProcAddress` for all of these CRT symbols from the Stan model DLL. The Stan model doesn't export them → error 127.

### Fix: Pre-generated bindings (no bindgen at build time)

We carry a patched copy of bridgestan 2.7.0 in `src/rust/patches/bridgestan-2.7.0/`. The patch has two changes from upstream:

1. **Pre-generated bindings**: `src/pregenerated_bindings.rs` contains the bindgen output for the 22 BridgeStan symbols, generated once with the correct allowlist filters. The `build.rs` simply copies this file to `OUT_DIR` — no bindgen runs at build time.

2. **No bindgen dependency**: `bindgen` is removed from `[build-dependencies]`, eliminating the need for LLVM/libclang on any platform.

The pre-generated bindings are platform-independent (they use `c_int`, `c_char`, `f64` etc. and `libloading` for dynamic dispatch).

Referenced via `[patch.crates-io]` in `Cargo.toml`.

**Upstream fix**: The root cause is arguably a bug in the bridgestan crate (missing allowlist filters). Consider filing a PR upstream — if merged, we could eventually drop the patch.

### Regenerating bindings

If bridgestan updates its C API, regenerate bindings by temporarily restoring bindgen in the patch's `Cargo.toml`, running a build, and copying the output from `$OUT_DIR/bindings.rs` back to `src/pregenerated_bindings.rs`.

## PATH and Tool Discovery

### Cargo/Rust

R's build subprocess doesn't inherit the user's full PATH on Windows. Two fixes were needed:

1. **`tools/msrv.R`**: Appends `$HOME/.cargo/bin` to PATH. Was using `:` as separator (Unix); fixed to use `;` on Windows.

2. **`Makevars.win.in`**: Exports `$(HOME)/.cargo/bin` on PATH so `make` can find `cargo`.

The Unix `Makevars.in` already had `PATH="$(PATH):$(HOME)/.cargo/bin"` inline on the cargo command.

### Make (for bridgestan::compile_model)

`bridgestan::compile_model()` shells out to `make` to compile Stan models. On Windows, rtools45 provides `make`. During `R CMD INSTALL`, rtools45 is on PATH (pkgbuild handles this). During interactive R sessions, it may or may not be.

`nutpie_compile_model()` checks for `make` on PATH and provides platform-specific installation guidance if missing.

## TBB (Threading Building Blocks)

Stan models compiled with `STAN_THREADS=true` dynamically link to `tbb.dll`. This DLL lives in the BridgeStan source tree at:
```
~/.bridgestan/bridgestan-X.Y.Z/stan/lib/stan_math/lib/tbb/
```

On macOS/Linux, `@rpath`/`RPATH` or `LD_LIBRARY_PATH` typically handles this. On Windows, the DLL must be on `PATH` at load time.

**Fix**: `add_tbb_to_path()` in `model.rs` scans `~/.bridgestan/` for TBB directories and prepends to `PATH` before loading the model. Uses `USERPROFILE` env var first (reliable on Windows), then `HOME`, then `dirs::home_dir()`.

## Spaces in Paths

The project lives under `Documents/R packages/nutpieR` — the space in `R packages` breaks:

- **dlltool/assembler**: Rust's import library generation splits `--temp-prefix` at spaces.
- **CARGO_HOME**: Must be quoted.

**Fix**: `Makevars.win.in` redirects `TARGET_DIR` and `CARGO_HOME` to `$(TEMP)/nutpieR-*` (no spaces). The Unix `Makevars.in` uses `$(CURDIR)` which is fine (no spaces in typical Unix paths).

## Debug vs Release Builds

Debug builds produce a ~269MB DLL that causes `pkgload::load_dll()` to fail (its `file.append` can't handle files that large). Always use release builds: `NOT_CRAN=true` with `devtools::install(quick = TRUE)`.

`devtools::load_all()` doesn't work well for this package — use `devtools::install()` instead.

## Build Artifact Size and Disk Space

Full release builds with LTO consume significant disk space. Current mitigations:
- `lto = "thin"` (faster than "fat", still decent optimization)
- `codegen-units = 4` (faster compile, slightly less optimization)
- Build artifacts go to `$(TEMP)/nutpieR-rust-target` and are cleaned after install

## Ideas for Simplifying

1. **Upstream the bridgestan fix**: File PR adding `allowlist_function`/`allowlist_var` to bridgestan's `build.rs`. If merged, we can drop the patch entirely.

2. **Consider a Makevars.ucrt** for the UCRT toolchain if R moves fully to UCRT (currently R 4.5 still uses MSVCRT via rtools45).
