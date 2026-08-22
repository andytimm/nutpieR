#![allow(non_snake_case)]

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeListArray,
    StringArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use extendr_api::error::Result;
use extendr_api::prelude::*;
use nuts_rs::{
    ArrowConfig, ArrowTrace, ChainProgress, DiagNutsSettings, LowRankNutsSettings,
    ProgressCallback, Sampler, SamplerWaitResult, Settings,
};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

extern "C" {
    static mut R_interrupts_pending: std::os::raw::c_int;
    static mut R_interrupts_suspended: std::os::raw::c_int;
    // Pump R's event loop. Front-ends (RStudio, Positron) buffer console output
    // produced during a blocking native call and only repaint when R processes
    // events, so without this the progress callback's output is invisible until
    // sample_stan() returns. See `pump_r_events`.
    fn R_ProcessEvents();
}

/// Flush front-end consoles mid-sampling by pumping R's event loop.
///
/// Interrupts are suspended across the call: on Unix `R_ProcessEvents` longjmps
/// (via `onintr`) when an interrupt is pending, which would skip the Rust
/// `Sampler` teardown and corrupt the in-flight result. Suspending lets it
/// flush output without jumping; the pending flag is left set and handled at the
/// top of the poll loop, which returns a clean error there instead.
fn pump_r_events() {
    unsafe {
        let prev = R_interrupts_suspended;
        R_interrupts_suspended = 1;
        R_ProcessEvents();
        R_interrupts_suspended = prev;
    }
}

/// Read and clear a pending R interrupt. Returns `true` if one was pending; the
/// caller should then return a clean `Err` so extendr raises a normal R error
/// at the call site ("Sampling interrupted.") rather than longjmp'ing past the
/// in-flight `Sampler` teardown.
fn interrupt_pending() -> bool {
    unsafe {
        if R_interrupts_pending != 0 {
            R_interrupts_pending = 0;
            true
        } else {
            false
        }
    }
}

mod model;

/// Convert any Display error to an extendr Error. Uses anyhow's alternate
/// Display format (`{:#}`) to preserve cause chains; a no-op for plain
/// `Display` impls that don't override the alternate flag.
fn r_err(e: impl std::fmt::Display) -> Error {
    Error::Other(format!("{e:#}"))
}

/// Validate a seed argument and convert to `u32`. Mirrors the R-side
/// `check_count(seed, max = .Machine$integer.max)` so direct callers
/// can't bypass it.
fn check_seed(seed: i32) -> Result<u32> {
    if seed < 0 {
        return Err(Error::Other(format!("seed must be >= 0, got {}", seed)));
    }
    Ok(seed as u32)
}

/// Unwrap an extendr `Result` at the FFI boundary, throwing a clean R error
/// on `Err` instead of going through extendr's panic-based conversion (which
/// prints `thread '<unnamed>' panicked...` to stderr before R catches it).
///
/// SAFETY note: `throw_r_error` longjmps and skips Rust destructors. Callers
/// must only invoke this from the outer `#[extendr]` boundary, after any
/// owned Rust state in the caller has been dropped — i.e. by delegating to
/// an `_impl` helper (or an immediately-invoked closure) that returns
/// `Result<T>` and calling `or_throw` on the returned `Result`.
fn or_throw<T>(r: Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => throw_r_error(e.to_string()),
    }
}

/// Return the linked BridgeStan crate version, e.g. "2.8.0". Used by the
/// inline-code compile cache key so a BridgeStan version bump invalidates
/// cached entries automatically.
/// @noRd
#[extendr]
fn bridgestan_version() -> String {
    bridgestan::VERSION.to_string()
}

/// Compile a Stan model to a shared library using BridgeStan.
/// Downloads BridgeStan sources if needed (first call is slow).
/// @param stan_file Path to the .stan file.
/// @param stanc_args Character vector of extra arguments for stanc compiler.
/// @param compile_args Character vector of extra arguments for make.
/// @return Path to the compiled shared library.
/// @noRd
#[extendr]
fn compile_stan_model(stan_file: &str, stanc_args: Strings, compile_args: Strings) -> String {
    or_throw(compile_stan_model_impl(stan_file, stanc_args, compile_args))
}

fn compile_stan_model_impl(
    stan_file: &str,
    stanc_args: Strings,
    compile_args: Strings,
) -> Result<String> {
    // Mirror the bridgestan crate's own cache-presence guard so the message
    // only fires on the genuine first-compile path. ~150s silent pause for
    // a 235 MB fetch was the single most reported dogfood surprise.
    if let Some(home) = dirs::home_dir() {
        let cache = home
            .join(".bridgestan")
            .join(format!("bridgestan-{}", bridgestan::VERSION));
        if !cache.exists() {
            rprintln!("Downloading BridgeStan sources (one-time, ~235 MB)...");
        }
    }
    let bs_path = bridgestan::download_bridgestan_src().map_err(r_err)?;
    // Make Stan's bundled macOS tbbmalloc_proxy safe to keep (GitHub #36).
    // No-op off macOS, when already patched, or when opted out via
    // NUTPIER_NO_TBB_PROXY_PATCH. `true`: a compile follows immediately, so it is
    // safe to purge and rebuild the proxy dylib.
    ensure_safe_tbb_proxy(&bs_path, true);
    let stan_path = PathBuf::from(stan_file);

    let stanc_vec: Vec<String> = if stanc_args.is_empty() {
        Vec::new()
    } else {
        stanc_args.iter().map(|s| s.to_string()).collect()
    };
    let stanc_refs: Vec<&str> = stanc_vec.iter().map(String::as_str).collect();
    let compile_vec: Vec<String> = if compile_args.is_empty() {
        Vec::new()
    } else {
        compile_args.iter().map(|s| s.to_string()).collect()
    };
    let compile_refs: Vec<&str> = compile_vec.iter().map(String::as_str).collect();

    let lib_path = bridgestan::compile_model(&bs_path, &stan_path, &stanc_refs, &compile_refs)
        .map_err(r_err)?;
    Ok(lib_path.to_string_lossy().into_owned())
}

// --- Issue #36: keep Stan's fast tbbmalloc_proxy allocator, made safe --------
//
// On macOS, Stan links `libtbbmalloc_proxy` into every model `.so`; it installs
// TBB as the process-wide malloc zone. libmalloc then calls TBB's zone `size()`
// callback for every `free()` in the process to find the owning zone, and that
// probe raw-reads the would-be block header just below the pointer. For a
// foreign pointer at a VM region start (e.g. a large R vector allocated before
// the model loaded) the read lands on an unmapped page and segfaults during R's
// GC — the crashes on GitHub #36. Dropping the proxy fixes it but costs ~17% on
// large, many-chain models. Instead we patch the proxy's zone callback to ask
// the other registered zones first whenever the probe would cross below a page
// boundary, and rebuild just the proxy dylib. The patched proxy also exports a
// marker symbol so the R layer can confirm at runtime that the *loaded* proxy is
// the safe one before it renders live progress (see `tbb_proxy_live_progress_safe`).

/// Sentinel string in the patched header; also the exported marker symbol name.
#[cfg(target_os = "macos")]
const TBB_PATCH_MARKER: &str = "nutpie_tbb_proxy_safe_probe";

/// Exact stock `impl_malloc_usable_size` from tbb_2020.3
/// `src/tbbmalloc/proxy_overload_osx.h`. Byte-stable across the BridgeStan
/// versions we bundle; if it ever changes, the patch declines cleanly and the
/// runtime gate takes over.
#[cfg(target_os = "macos")]
const TBB_STOCK_FN: &str = r#"/* note: impl_malloc_usable_size() is called for each free() call, so it must be fast */
static size_t impl_malloc_usable_size(struct _malloc_zone_t *, const void *ptr)
{
    // malloc_usable_size() is used by macOS* to recognize which memory manager
    // allocated the address, so our wrapper must not redirect to the original function.
    return __TBB_malloc_safer_msize(const_cast<void*>(ptr), NULL);
}"#;

/// Replacement: an exported marker symbol plus a page-boundary-guarded callback.
/// The guard is two integer compares on the hot path. How often the guarded
/// slow path fires depends on the VM page size: on 16K-page Apple Silicon it is
/// essentially never (~0.1%, only pointers hugging a page boundary via
/// `headerMayCrossPage`); on 4K-page Intel it is ~75% of frees, because any
/// pointer whose 16K slab floor sits below its 4K page trips `slabFloorBelowPage`.
/// The slow path walks the other registered zones from `malloc_get_all_zones`.
/// It must NOT shortcut via `malloc_default_zone()`: that returns libmalloc's
/// virtual-default-zone wrapper, and the proxy registers its own zone as zone 0
/// (the runtime default — see the system-zone unregister/re-register dance in
/// proxy_overload_osx.h), so the wrapper's `size()` forwards straight back into
/// this callback and recurses without bound.
#[cfg(target_os = "macos")]
const TBB_PATCHED_FN: &str = r#"/* nutpieR (GitHub #36): exported marker so the R layer can confirm at runtime,
   via dlsym, that the loaded tbbmalloc_proxy carries the page-boundary-safe zone
   size() probe below. A proxy loaded WITHOUT this symbol gates off live progress
   to avoid the GC-time segfault in __TBB_malloc_safer_msize. */
extern "C" int nutpie_tbb_proxy_safe_probe = 1;

/* note: impl_malloc_usable_size() is called for each free() call, so it must be fast */
static size_t impl_malloc_usable_size(struct _malloc_zone_t *self, const void *ptr)
{
    // malloc_usable_size() is used by macOS* to recognize which memory manager
    // allocated the address, so our wrapper must not redirect to the original function.
    //
    // Safety (nutpieR #36): __TBB_malloc_safer_msize decides whether a block is
    // ours by dereferencing the would-be large-object header just below `ptr`
    // (16 bytes) and the would-be slab header at alignDown(ptr, 16K). For a
    // foreign pointer at the very start of a VM region (e.g. a libmalloc
    // MALLOC_LARGE block allocated before this zone was installed), those
    // addresses can lie on an unmapped page and the probe itself raises
    // EXC_BAD_ACCESS. Only pointers whose probe addresses cross below a page
    // boundary are at risk; for those, ask the other registered zones first --
    // if one claims the pointer it is certainly not ours, and we must return 0
    // without touching the (possibly unmapped) memory below it.
    {
        const uintptr_t p = (uintptr_t)ptr;
        // vm_page_mask is a libsystem global (via <mach/mach.h>), initialized
        // before user code runs -- a plain load, unlike a function-local
        // runtime-init `static` which would cost a __cxa_guard atomic check on
        // every free(). On 4K-page Intel this guard fires for ~75% of frees, so
        // it is on a genuinely hot path.
        bool headerMayCrossPage = (p & (uintptr_t)vm_page_mask) < 16;    /* sizeof(LargeObjectHdr) */
        bool slabFloorBelowPage = (p & ~(uintptr_t)(16*1024 - 1)) < (p & ~(uintptr_t)vm_page_mask);
        if (headerMayCrossPage || slabFloorBelowPage) {
            // Walk the registered zones (skipping ourselves) and let the real
            // owner claim the pointer. Do NOT shortcut by probing libmalloc's
            // default-zone accessor: it returns the virtual-default-zone
            // wrapper, and this zone IS the runtime default (zone 0, see the
            // registration dance below), so the wrapper's size() would forward
            // right back here and recurse. The zones array below holds only
            // concrete zones, so `z != self` is a sufficient recursion guard.
            // Do NOT cache the zone list in a static -- zones can register
            // after this runs.
            vm_address_t *zones = NULL;
            unsigned count = 0;
            if (KERN_SUCCESS == malloc_get_all_zones(mach_task_self(), NULL, &zones, &count)) {
                for (unsigned i = 0; i < count; ++i) {
                    malloc_zone_t *z = (malloc_zone_t*)zones[i];
                    if (z && z != self && z->size(z, ptr) > 0)
                        return 0;
                }
            }
        }
    }
    return __TBB_malloc_safer_msize(const_cast<void*>(ptr), NULL);
}"#;

/// Splice the safe callback into the proxy source. Returns `None` if the stock
/// function is not found verbatim (unfamiliar TBB version) so the caller can
/// decline rather than corrupt the file.
#[cfg(target_os = "macos")]
fn patch_proxy_source(content: &str) -> Option<String> {
    if !content.contains(TBB_STOCK_FN) {
        return None;
    }
    Some(content.replacen(TBB_STOCK_FN, TBB_PATCHED_FN, 1))
}

/// Patch Stan's bundled tbbmalloc_proxy source and (optionally) force a rebuild
/// of the proxy dylib. Idempotent; best-effort (any IO problem leaves the source
/// untouched and the runtime gate takes over). macOS-only.
///
/// `will_recompile` MUST be true only when a model compile immediately follows
/// (the compile path), because purging the artifacts deletes
/// `libtbbmalloc_proxy.dylib`, which every model `.so` hard-links via @rpath
/// (`LC_LOAD_DYLIB`). The compile rebuilds it before linking, so that is safe.
/// From the cache-hit path no compile follows, so we pass `false`: we still
/// patch the header (so the *next* real compile produces the safe proxy) but
/// never delete the dylib — deleting it would break the cached model that is
/// about to be loaded. Until that next compile, the current session is covered
/// by the runtime gate (`tbb_proxy_live_progress_safe`).
#[cfg(target_os = "macos")]
fn ensure_safe_tbb_proxy(bs_path: &std::path::Path, will_recompile: bool) {
    use std::fs;
    // Opt out only on exactly "1", matching the sibling
    // NUTPIER_DISABLE_COMPILE_CACHE convention (R/compile.R) and the documented
    // `=1` in NEWS/help -- not on any arbitrary value.
    if std::env::var("NUTPIER_NO_TBB_PROXY_PATCH").as_deref() == Ok("1") {
        return;
    }
    let lib = bs_path
        .join("stan")
        .join("lib")
        .join("stan_math")
        .join("lib");
    // Stan vendors TBB under lib/tbb_<version>/; the built objects and dylibs
    // land in lib/tbb/.
    let src_dir = match fs::read_dir(&lib) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("tbb_"))
                && p.join("src/tbbmalloc/proxy_overload_osx.h").exists()
        }),
        Err(_) => None,
    };
    let src_dir = match src_dir {
        Some(d) => d,
        None => return,
    };
    let header = src_dir
        .join("src")
        .join("tbbmalloc")
        .join("proxy_overload_osx.h");
    let content = match fs::read_to_string(&header) {
        Ok(c) => c,
        Err(_) => return,
    };
    let bin = lib.join("tbb");
    if content.contains(TBB_PATCH_MARKER) {
        // The header is already patched, but a marker in the *source* only proves
        // the header was written -- not that the dylib was rebuilt from it. If a
        // prior run patched the header but the artifact purge failed (or the
        // session died between the write and the purge), the stale unpatched
        // dylib keeps shipping and every future call would early-return here
        // forever. Re-verify against the built dylib: if it exists but its bytes
        // don't carry the marker symbol, purge the artifacts so the next model
        // build rebuilds it (idempotent, cheap). Only when a recompile follows —
        // purging deletes the dylib, and off the compile path nothing rebuilds it.
        if will_recompile {
            let dylib = bin.join("libtbbmalloc_proxy.dylib");
            if let Ok(bytes) = fs::read(&dylib) {
                if !contains_bytes(&bytes, TBB_PATCH_MARKER.as_bytes()) {
                    purge_tbb_proxy_artifacts(&bin);
                }
            }
        }
        return;
    }
    let patched = match patch_proxy_source(&content) {
        Some(p) => p,
        None => {
            rprintln!(
                "nutpieR: bundled tbbmalloc_proxy source is not in the expected form; \
                 leaving it unpatched and gating live progress at runtime (GitHub #36)."
            );
            return;
        }
    };
    if fs::write(&header, patched).is_err() {
        return;
    }
    // Force a rebuild of the proxy dylib from the patched source — but only when
    // a compile follows to actually rebuild it. On the cache-hit path the header
    // is now patched so the next real compile produces the safe proxy; deleting
    // the dylib here would break the cached model about to be loaded.
    if will_recompile {
        purge_tbb_proxy_artifacts(&bin);
    }
}

/// Force a rebuild of the proxy dylib from the (patched) source. In Stan's make
/// graph (make/libraries) the `libtbbmalloc_proxy.dylib` target has NO recipe of
/// its own — the dylib is produced only as a side effect of the `tbbmalloc.def`
/// recipe, which re-runs only when that `.def` is missing. So removing just the
/// dylib leaves make unable to rebuild it and the next model link fails. Remove
/// the `.def` (the real trigger) plus the stale object and dylib; the next model
/// build then recompiles proxy.cpp — the only TU that includes this header —
/// from the patched source. The proxy dylib links with no export filter, so the
/// marker symbol becomes dlsym-visible, and models resolve it via @rpath at load
/// time, so even previously compiled models pick up the safe proxy in a fresh
/// session. Idempotent and cheap.
#[cfg(target_os = "macos")]
fn purge_tbb_proxy_artifacts(bin: &std::path::Path) {
    use std::fs;
    let _ = fs::remove_file(bin.join("tbbmalloc.def"));
    let _ = fs::remove_file(bin.join("proxy.o"));
    let _ = fs::remove_file(bin.join("libtbbmalloc_proxy.dylib"));
}

/// Simple substring search over raw bytes (the marker symbol name appears in the
/// dylib's symbol table).
#[cfg(target_os = "macos")]
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return needle.is_empty();
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(not(target_os = "macos"))]
fn ensure_safe_tbb_proxy(_bs_path: &std::path::Path, _will_recompile: bool) {}

/// Cache-hit companion to the compile-time patch. `compile_via_cache()` (R)
/// returns a cached model without ever calling into Rust, so an upgrading user
/// whose model is already cached would never re-run `ensure_safe_tbb_proxy` and
/// would keep shipping the stale, unpatched dylib. This re-applies the patch,
/// but ONLY when the BridgeStan source tree already exists on disk — it must
/// never trigger the ~235 MB download on a cache hit. macOS-cheap, idempotent,
/// and a no-op elsewhere.
/// @noRd
#[extendr]
fn ensure_tbb_proxy_patched() {
    #[cfg(target_os = "macos")]
    {
        // Same path the bridgestan crate resolves to (see the download guard in
        // compile_stan_model_impl); only touch it if it already exists so we
        // never kick off a network download from the cache-hit path.
        if let Some(home) = dirs::home_dir() {
            let bs_path = home
                .join(".bridgestan")
                .join(format!("bridgestan-{}", bridgestan::VERSION));
            if bs_path.exists() {
                // No compile follows a cache hit, so pass `false`: patch the
                // header (the next real compile then builds the safe proxy) but
                // never delete the proxy dylib the cached model links against.
                ensure_safe_tbb_proxy(&bs_path, false);
            }
        }
    }
}

/// Runtime companion to the compile-time patch: is it safe to render live
/// progress (which allocates R memory mid-sample and can trigger GC)?
///
/// Safe when no `tbbmalloc_proxy` is loaded in the process, or when the loaded
/// one exports our marker symbol (i.e. it is the patched, page-safe build). The
/// only unsafe case is a stale, unpatched proxy already loaded in this session —
/// e.g. a model compiled before the patch and loaded in the same R session. The
/// R layer downgrades live progress to a static summary in that case (#36).
#[cfg(target_os = "macos")]
mod tbb_gate {
    use std::os::raw::c_char;

    // libc marks the _dyld_* image-iteration functions deprecated (it points at
    // the mach2 crate), but they are the documented libdyld API and remain the
    // lightest way to answer "is this image loaded?" without a new dependency.
    #[allow(deprecated)]
    pub fn live_progress_safe() -> bool {
        unsafe {
            let mut proxy_loaded = false;
            for i in 0..libc::_dyld_image_count() {
                let name = libc::_dyld_get_image_name(i);
                if name.is_null() {
                    continue;
                }
                if std::ffi::CStr::from_ptr(name)
                    .to_string_lossy()
                    .contains("libtbbmalloc_proxy")
                {
                    proxy_loaded = true;
                    break;
                }
            }
            if !proxy_loaded {
                return true;
            }
            // libc::RTLD_DEFAULT resolves the symbol in any loaded image.
            let sym = b"nutpie_tbb_proxy_safe_probe\0";
            !libc::dlsym(libc::RTLD_DEFAULT, sym.as_ptr() as *const c_char).is_null()
        }
    }
}

/// Whether nutpieR's live progress renderer is safe to run given the currently
/// loaded allocator (GitHub #36). Always TRUE off macOS.
/// @noRd
#[extendr]
fn tbb_proxy_live_progress_safe() -> bool {
    #[cfg(target_os = "macos")]
    {
        tbb_gate::live_progress_safe()
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Expose the stock TBB function text and the nutpieR marker to R so a test can
/// assert the bundled proxy header still matches one of them — a BridgeStan/TBB
/// bump that broke the verbatim splice would then fail loudly instead of
/// silently reverting macOS users to the unpatched proxy (GitHub #36). Returns
/// `c(stock, marker)` on macOS, `character(0)` elsewhere.
/// @noRd
#[extendr]
fn tbb_patch_strings() -> Vec<String> {
    #[cfg(target_os = "macos")]
    {
        vec![TBB_STOCK_FN.to_string(), TBB_PATCH_MARKER.to_string()]
    }
    #[cfg(not(target_os = "macos"))]
    {
        Vec::new()
    }
}

#[derive(Clone, Default)]
struct ChainState {
    chain: usize,
    finished_draws: usize,
    total_draws: usize,
    divergences: usize,
    tuning: bool,
    started: bool,
    latest_num_steps: usize,
    total_num_steps: usize,
    step_size: f64,
    runtime: Duration,
    divergent_draws: Vec<usize>,
}

/// Sample from a Stan model using nuts-rs NUTS sampler.
/// Run the sampler with progress reporting. Generic over Settings type.
fn run_sampler<S: Settings>(
    stan_model: model::StanModel,
    settings: S,
    num_cores: i32,
    save_warmup: bool,
    progress_cb: Option<Function>,
) -> Result<Vec<ArrowTrace>> {
    let mut progress_cb = progress_cb;
    let use_callback = progress_cb.is_some();

    let progress_state: Arc<Mutex<Vec<ChainState>>> = Arc::new(Mutex::new(Vec::new()));
    let state_clone = progress_state.clone();

    // Only install the nuts-rs progress callback when R wants per-poll
    // snapshots. With `progress = "none"` there is no R callback, so we skip
    // the snapshot bookkeeping — but the wait loop below still runs to service
    // interrupts.
    let callback = if use_callback {
        Some(ProgressCallback {
            callback: Box::new(move |_elapsed: Duration, progress: Box<[ChainProgress]>| {
                let snapshot: Vec<ChainState> = progress
                    .iter()
                    .enumerate()
                    .map(|(i, c)| ChainState {
                        chain: i + 1,
                        finished_draws: c.finished_draws,
                        total_draws: c.total_draws,
                        divergences: c.divergences,
                        tuning: c.tuning,
                        started: c.started,
                        latest_num_steps: c.latest_num_steps,
                        total_num_steps: c.total_num_steps,
                        step_size: c.step_size,
                        runtime: c.runtime,
                        divergent_draws: c.divergent_draws.clone(),
                    })
                    .collect();
                *state_clone.lock().unwrap() = snapshot;
            }),
            rate: Duration::from_millis(100),
        })
    } else {
        None
    };

    let mut arrow_config = ArrowConfig::default();
    arrow_config.store_warmup = save_warmup;
    let mut sampler_opt = Some(
        Sampler::new(
            stan_model,
            settings,
            arrow_config,
            num_cores as usize,
            callback,
        )
        .map_err(r_err)?,
    );

    // Interrupt teardown, written once for the two poll sites below (a macro
    // rather than a closure/fn only because `Sampler`'s concrete generic type is
    // not nameable here). nuts-rs `Sampler` has no Drop impl, so simply dropping
    // it only detaches the rayon pool — the worker threads keep finalizing in the
    // background into R's post-interrupt GC, and the two then push concurrent
    // allocator traffic that can segfault during GC (#36). abort() signals the
    // controller (drop of the command channel) and joins the threads before we
    // unwind, then extendr raises a clean R error at the call site.
    macro_rules! abort_on_interrupt {
        ($opt:expr) => {{
            if let Some(sampler) = $opt.take() {
                let _ = sampler.abort();
            }
            return Err(Error::Other("Sampling interrupted.".into()));
        }};
    }

    let results = loop {
        let sampler = sampler_opt.take().unwrap();
        let wait_dur = Duration::from_millis(200);
        match sampler.wait_timeout(wait_dur) {
            SamplerWaitResult::Trace(traces) => break traces,
            SamplerWaitResult::Timeout(s) => {
                sampler_opt = Some(s);
                // Check interrupt BEFORE calling back into R, so extendr raises a
                // clean R error at the call site ("Sampling interrupted.") rather
                // than longjmp'ing past the assignment and leaving the user with a
                // confusing "object not found" on the next line.
                if interrupt_pending() {
                    abort_on_interrupt!(sampler_opt);
                }
                // Repaint the front-end console each poll so progress streams
                // live instead of appearing all at once when sampling ends
                // (GitHub #34: RStudio buffers native-call output).
                pump_r_events();
                // pump_r_events() suspends interrupts, so R_ProcessEvents() can
                // latch a pending interrupt without longjmp'ing. Re-check before
                // the R progress callback runs, or it could service the flag by
                // longjmp'ing across these Rust frames.
                if interrupt_pending() {
                    abort_on_interrupt!(sampler_opt);
                }
                let state_snapshot: Vec<ChainState> = {
                    let state = progress_state.lock().unwrap();
                    state.clone()
                };
                if state_snapshot.is_empty() {
                    continue;
                }

                let cb_failed = if let Some(ref cb) = progress_cb {
                    let snapshot_list = build_progress_snapshot(&state_snapshot);
                    let pairlist = Pairlist::from_pairs([("", Robj::from(snapshot_list))]);
                    match cb.call(pairlist) {
                        Ok(_) => false,
                        Err(e) => {
                            rprintln!(
                                "nutpieR: progress callback failed ({}); disabling further callbacks for this run.",
                                e
                            );
                            true
                        }
                    }
                } else {
                    false
                };
                if cb_failed {
                    progress_cb = None;
                }
            }
            // Residual unjoined-threads race (#36): nuts-rs 0.18.2's wait_timeout
            // Ok(Err(e)) path drops the `Sampler` here — dropping the command
            // channel and the JoinHandle *unjoined* — so worker threads can still
            // be finalizing while R unwinds and runs GC, the same hazard the
            // interrupt path guards against. We cannot guard it here: this arm
            // hands back only the error, not the sampler, so there is no handle to
            // abort()/join. Fixing it needs upstream nuts-rs support (a Drop impl
            // on Sampler, or an Err variant that carries the sampler back).
            SamplerWaitResult::Err(e, _) => return Err(r_err(e)),
        }
    };

    if let Some(ref cb) = progress_cb {
        let state_snapshot: Vec<ChainState> = {
            let state = progress_state.lock().unwrap();
            state.clone()
        };
        if !state_snapshot.is_empty() {
            let snapshot_list = build_progress_snapshot(&state_snapshot);
            let pairlist = Pairlist::from_pairs([("", Robj::from(snapshot_list))]);
            let _ = cb.call(pairlist);
        }
    }

    Ok(results)
}

/// Build a `List` of per-chain named lists for the R progress callback.
/// Field set must match the R-side progress helpers in `R/progress.R`.
fn build_progress_snapshot(state: &[ChainState]) -> List {
    let entries: Vec<Robj> = state
        .iter()
        .map(|c| {
            list!(
                chain = c.chain as i32,
                finished_draws = c.finished_draws as i32,
                total_draws = c.total_draws as i32,
                divergences = c.divergences as i32,
                tuning = c.tuning,
                started = c.started,
                latest_num_steps = c.latest_num_steps as i32,
                total_num_steps = c.total_num_steps as f64,
                step_size = c.step_size,
                runtime = c.runtime.as_secs_f64(),
                divergent_draws = c
                    .divergent_draws
                    .iter()
                    .map(|x| *x as i32)
                    .collect::<Vec<i32>>()
            )
            .into_robj()
        })
        .collect();
    List::from_values(entries)
}

/// @param handle An `ExternalPtr<BSHandle>` from `bs_open()`.
/// @param num_draws Number of draws per chain after warmup.
/// @param num_warmup Number of warmup (tuning) draws per chain.
/// @param num_chains Number of parallel chains.
/// @param seed Random seed.
/// @param init_positions Optional list of numeric vectors (one per chain, or length 1 = broadcast).
/// @param jitter If TRUE, apply ±0.5 uniform jitter per coordinate.
/// @param save_warmup Whether to return warmup draws.
/// @param num_cores Number of CPU cores to use for parallel sampling.
/// @param store_divergences Whether to store detailed divergence information.
/// @param store_mass_matrix Whether to store the mass matrix at each draw.
/// @param store_unconstrained Whether to store the unconstrained position at each draw.
/// @param store_gradient Whether to store the gradient at each draw.
/// @param adaptation One of "diag" or "low_rank". The R wrapper accepts
///   "low-rank" as a Python-style alias and normalises it before calling.
/// @param max_treedepth Optional maximum tree depth for NUTS. NULL keeps the
///   nuts-rs default.
/// @param mindepth Optional minimum tree depth for NUTS.
/// @param target_accept Optional target acceptance probability.
/// @param max_energy_error Optional energy-error divergence threshold.
/// @param extra_doublings Optional number of extra tree doublings after a
///   turning point is reached.
/// @param mass_matrix_gamma Optional regularisation parameter for low-rank
///   mass matrix.
/// @param eigval_cutoff Optional eigenvalue cutoff for low-rank mass matrix.
/// @param keep_indices Optional 0-indexed integer vector of constrained
///   parameter columns to materialize. NULL means keep all. Indices are
///   resolved against the post-flag column layout selected by
///   `include_tp` / `include_gq`.
/// @param include_tp Whether bridgestan should compute transformed parameters
///   when expanding each draw. When the caller has filtered them out via
///   `pars`/`include`, set this to `FALSE` to skip the per-draw allocation
///   and Stan-side work.
/// @param include_gq Whether bridgestan should compute generated quantities
///   when expanding each draw. Setting this `FALSE` skips the GQ block
///   (including any `*_rng` calls) entirely. Must imply `include_tp = TRUE`
///   when `TRUE`, since GQ may reference TP.
/// @param progress_callback NULL or an R closure invoked on each poll wakeup
///   with one argument: a list of `num_chains` per-chain snapshots
///   (`chain`, `finished_draws`, `total_draws`, `divergences`, `tuning`,
///   `started`, `latest_num_steps`, `total_num_steps`, `step_size`, `runtime`,
///   `divergent_draws`). When supplied, the built-in per-chain text log is
///   suppressed; errors raised by the closure are warned once and further calls
///   suppressed for the run.
/// @return A named list with draws matrix, num_warmup, num_chains, diagnostics,
///   sampler_config (JSON), and optionally warmup_draws and warmup_diagnostics.
/// @noRd
#[allow(clippy::too_many_arguments)]
#[extendr]
fn sample_stan(
    handle: ExternalPtr<model::BSHandle>,
    num_draws: i32,
    num_warmup: i32,
    num_chains: i32,
    seed: i32,
    init_positions: Robj,
    jitter: bool,
    save_warmup: bool,
    num_cores: i32,
    store_divergences: bool,
    store_mass_matrix: bool,
    store_unconstrained: bool,
    store_gradient: bool,
    adaptation: &str,
    max_treedepth: Robj,
    mindepth: Robj,
    target_accept: Robj,
    max_energy_error: Robj,
    extra_doublings: Robj,
    mass_matrix_gamma: Robj,
    eigval_cutoff: Robj,
    keep_indices: Robj,
    include_tp: bool,
    include_gq: bool,
    progress_callback: Robj,
) -> List {
    or_throw((|| -> Result<List> {
        // Defensive guards before unsigned casts. The R wrapper validates these
        // already; we re-check here so direct callers (or malformed inputs that
        // somehow slip past the R layer) can't turn negative ints into huge
        // u64/usize allocations.
        for (val, name) in [
            (num_draws, "num_draws"),
            (num_chains, "num_chains"),
            (num_cores, "num_cores"),
        ] {
            if val <= 0 {
                return Err(Error::Other(format!("{} must be >= 1, got {}", name, val)));
            }
        }
        if num_warmup < 0 {
            return Err(Error::Other(format!(
                "num_warmup must be >= 0, got {}",
                num_warmup
            )));
        }
        check_seed(seed)?;

        let max_treedepth_opt = opt_count(&max_treedepth, "max_treedepth", 1)?;
        let mindepth_opt = opt_count(&mindepth, "mindepth", 0)?;
        let target_accept_opt = opt_finite_in_open_unit(&target_accept, "target_accept")?;
        let max_energy_error_opt = opt_finite_positive_f64(&max_energy_error, "max_energy_error")?;
        let extra_doublings_opt = opt_count(&extra_doublings, "extra_doublings", 0)?;

        let init_positions_raw: Option<Vec<Vec<f64>>> = if init_positions.is_null() {
            None
        } else {
            let lst = init_positions.as_list().ok_or_else(|| {
                Error::Other("init_positions must be a list of numeric vectors".into())
            })?;
            let mut out = Vec::with_capacity(lst.len());
            for (i, (_, el)) in lst.iter().enumerate() {
                let v = el.as_real_vector().ok_or_else(|| {
                    Error::Other(format!(
                        "init_positions[[{}]] must be a numeric vector",
                        i + 1
                    ))
                })?;
                out.push(v);
            }
            Some(out)
        };

        let stan_model = model::StanModel::new(&handle)
            .with_init_positions(init_positions_raw, jitter)
            .and_then(|m| m.with_constrain_flags(&handle, include_tp, include_gq))
            .map_err(r_err)?;

        let ndim = stan_model.num_constrained();
        let all_param_names: &[String] = stan_model.constrained_param_names();

        // Resolve keep_indices: NULL → keep all, else use as-supplied. Indices
        // are 0-based and must be within [0, ndim).
        let keep_cols: Vec<usize> = if keep_indices.is_null() {
            (0..ndim).collect()
        } else {
            let v = keep_indices.as_integer_vector().ok_or_else(|| {
                Error::Other("keep_indices must be NULL or an integer vector".into())
            })?;
            let mut out = Vec::with_capacity(v.len());
            for &idx in v.iter() {
                if idx < 0 || (idx as usize) >= ndim {
                    return Err(Error::Other(format!(
                        "keep_indices contains out-of-range value {} (ndim={})",
                        idx, ndim
                    )));
                }
                out.push(idx as usize);
            }
            out
        };
        let kept_param_names: Vec<String> = keep_cols
            .iter()
            .map(|&i| all_param_names[i].clone())
            .collect();
        let expand_error_count = stan_model.expand_error_count_handle();

        let num_tune = num_warmup as usize;
        let n_draws_per_chain = num_draws as usize;

        macro_rules! configure_settings {
            ($settings:expr) => {{
                $settings.num_tune = num_warmup as u64;
                $settings.num_draws = num_draws as u64;
                $settings.num_chains = num_chains as usize;
                $settings.seed = seed as u64;
                if let Some(v) = max_treedepth_opt {
                    $settings.maxdepth = v as u64;
                }
                if let Some(v) = mindepth_opt {
                    $settings.mindepth = v as u64;
                }
                if let Some(v) = target_accept_opt {
                    $settings.adapt_options.step_size_settings.target_accept = v;
                }
                if let Some(v) = max_energy_error_opt {
                    $settings.max_energy_error = v;
                }
                if let Some(v) = extra_doublings_opt {
                    $settings.extra_doublings = v as u64;
                }
                $settings.store_divergences = store_divergences;
                $settings.store_unconstrained = store_unconstrained;
                $settings.store_gradient = store_gradient;
                $settings
                    .adapt_options
                    .mass_matrix_options
                    .store_mass_matrix = store_mass_matrix;
            }};
        }

        let progress_callback = if progress_callback.is_null() {
            None
        } else {
            Some(progress_callback.as_function().ok_or_else(|| {
                Error::Other("progress_callback must be NULL or a function".into())
            })?)
        };

        let (results, sampler_config_json) = match adaptation {
            "low_rank" => {
                let mut settings = LowRankNutsSettings::default();
                configure_settings!(settings);
                if let Some(v) = opt_finite_positive_f64(&mass_matrix_gamma, "mass_matrix_gamma")? {
                    settings.adapt_options.mass_matrix_options.gamma = v;
                }
                if let Some(v) = opt_finite_positive_f64(&eigval_cutoff, "eigval_cutoff")? {
                    settings.adapt_options.mass_matrix_options.eigval_cutoff = v;
                }
                run_with_settings(
                    stan_model,
                    settings,
                    num_cores,
                    save_warmup,
                    progress_callback.clone(),
                )?
            }
            "diag" => {
                let mut settings = DiagNutsSettings::default();
                configure_settings!(settings);
                run_with_settings(
                    stan_model,
                    settings,
                    num_cores,
                    save_warmup,
                    progress_callback.clone(),
                )?
            }
            other => {
                return Err(Error::Other(format!(
                    "adaptation must be one of \"diag\" or \"low_rank\", got \"{}\"",
                    other
                )));
            }
        };

        let post_warmup_skip = if save_warmup { num_tune } else { 0 };

        let draws_robj = build_draws_matrix(
            &results,
            &keep_cols,
            post_warmup_skip,
            n_draws_per_chain,
            &kept_param_names,
        )?;

        // Defensive boundary filter for the two opt-in, `ndim_unc`-wide
        // diagnostics. The pinned nuts-rs honours both storage flags, so these
        // columns are normally all-null and the schema-driven filter below
        // would drop them anyway. Keep the explicit suppression so the R API
        // continues to honour its defaults if an upstream schema or settings
        // regression ever populates them unexpectedly.
        let mut drop_cols: Vec<&str> = Vec::new();
        if !store_unconstrained {
            drop_cols.push("unconstrained_draw");
        }
        if !store_gradient {
            drop_cols.push("gradient");
        }

        let diagnostics =
            extract_diagnostics(&results, post_warmup_skip, n_draws_per_chain, &drop_cols)?;

        let warmup_draws_robj: Robj = if save_warmup {
            build_draws_matrix(&results, &keep_cols, 0, num_tune, &kept_param_names)?
        } else {
            ().into_robj()
        };
        let warmup_diagnostics_robj: Robj = if save_warmup {
            extract_diagnostics(&results, 0, num_tune, &drop_cols)?.into_robj()
        } else {
            ().into_robj()
        };

        let n_expand_errors = expand_error_count.load(Ordering::Relaxed) as i32;

        Ok(list!(
            draws = draws_robj,
            num_warmup = num_warmup,
            num_chains = num_chains,
            diagnostics = diagnostics,
            warmup_draws = warmup_draws_robj,
            warmup_diagnostics = warmup_diagnostics_robj,
            expand_errors = n_expand_errors,
            sampler_config = sampler_config_json
        ))
    })())
}

/// Optional finite scalar (`NULL` -> None, REAL scalar -> Some). Caller does
/// any range checks. Mirrors the R-side `check_count` so direct FFI callers
/// can't bypass validation.
fn opt_finite_f64(robj: &Robj, name: &str) -> Result<Option<f64>> {
    if robj.is_null() {
        return Ok(None);
    }
    let v: f64 = robj
        .as_real()
        .ok_or_else(|| Error::Other(format!("`{}` must be NULL or a single numeric.", name)))?;
    if !v.is_finite() {
        return Err(Error::Other(format!("`{}` must be a finite number.", name)));
    }
    Ok(Some(v))
}

fn opt_finite_positive_f64(robj: &Robj, name: &str) -> Result<Option<f64>> {
    let v = opt_finite_f64(robj, name)?;
    if let Some(x) = v {
        if x <= 0.0 {
            return Err(Error::Other(format!("`{}` must be > 0, got {}.", name, x)));
        }
    }
    Ok(v)
}

fn opt_finite_in_open_unit(robj: &Robj, name: &str) -> Result<Option<f64>> {
    let v = opt_finite_f64(robj, name)?;
    if let Some(x) = v {
        if x <= 0.0 || x >= 1.0 {
            return Err(Error::Other(format!(
                "`{}` must be in (0, 1), got {}.",
                name, x
            )));
        }
    }
    Ok(v)
}

fn opt_count(robj: &Robj, name: &str, min: i32) -> Result<Option<i32>> {
    let Some(v) = opt_finite_f64(robj, name)? else {
        return Ok(None);
    };
    if v.fract() != 0.0 {
        return Err(Error::Other(format!(
            "`{}` must be a whole number, got {}.",
            name, v
        )));
    }
    if v < min as f64 || v > i32::MAX as f64 {
        return Err(Error::Other(format!(
            "`{}` must be in [{}, {}], got {}.",
            name,
            min,
            i32::MAX,
            v
        )));
    }
    Ok(Some(v as i32))
}

/// Run the sampler with `settings` and return the traces alongside a JSON
/// snapshot of the effective settings (surfaced via `attr(draws, "sampler_config")`).
fn run_with_settings<S: Settings + serde::Serialize>(
    stan_model: model::StanModel,
    settings: S,
    num_cores: i32,
    save_warmup: bool,
    progress_callback: Option<Function>,
) -> Result<(Vec<ArrowTrace>, String)> {
    let json = serde_json::to_string(&settings)
        .map_err(|e| Error::Other(format!("failed to serialize sampler settings: {}", e)))?;
    let traces = run_sampler(
        stan_model,
        settings,
        num_cores,
        save_warmup,
        progress_callback,
    )?;
    Ok((traces, json))
}

/// Build a draws matrix from Arrow traces.
///
/// Materializes only the columns listed in `keep_cols` (0-indexed, against
/// the full constrained parameter dimension), writing directly into an
/// R-allocated `Doubles` in column-major order. On wide models with
/// large transformed-parameter / generated-quantities blocks this avoids
/// allocating columns the user is going to drop anyway.
///
/// `skip` is the number of initial Arrow rows to skip, `n_draws` is how
/// many to extract. `param_names` must already be filtered to match
/// `keep_cols`.
fn build_draws_matrix(
    results: &[ArrowTrace],
    keep_cols: &[usize],
    skip: usize,
    n_draws: usize,
    param_names: &[String],
) -> Result<Robj> {
    let n_chains = results.len();
    let total_rows = n_draws * n_chains;
    let n_kept = keep_cols.len();

    let mut out = Doubles::new(total_rows * n_kept);
    let dest: &mut [Rfloat] = &mut out;

    for (chain_idx, trace) in results.iter().enumerate() {
        let batch = &trace.posterior;
        let col = batch
            .column_by_name("value")
            .ok_or_else(|| Error::Other("No 'value' column in posterior".into()))?;

        let list_arr = col
            .as_any()
            .downcast_ref::<arrow::array::LargeListArray>()
            .ok_or_else(|| Error::Other("'value' column is not LargeList".into()))?;

        for draw in 0..n_draws {
            let row = skip + draw;
            let inner = list_arr.value(row);
            let values = inner
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| Error::Other("inner array is not Float64".into()))?;
            let row_data = values.values();

            let dest_row = chain_idx * n_draws + draw;
            for (out_col, &src_col) in keep_cols.iter().enumerate() {
                dest[dest_row + out_col * total_rows] = Rfloat::from(row_data[src_col]);
            }
        }
    }

    let mut robj: Robj = out.into();
    robj.set_attrib("dim", [total_rows as i32, n_kept as i32].into_robj())
        .map_err(r_err)?;
    if !param_names.is_empty() {
        let colnames: Vec<&str> = param_names.iter().map(|s| s.as_str()).collect();
        let dimnames = List::from_values(&[
            ().into_robj(), // rownames = NULL
            colnames.into_robj(),
        ]);
        robj.set_attrib("dimnames", dimnames).ok();
    }
    Ok(robj)
}

/// Convert an Arrow column window (across all chains) into an R object.
///
/// `Float*` columns always become R `double`. `Int*`/`UInt*` columns become R
/// `integer` when every non-null value fits in `(i32::MIN, i32::MAX]`
/// (`i32::MIN` is reserved as `NA_INTEGER`); otherwise they fall back to
/// `double`. `logp` is `Float64` in the nuts-rs schema, so it can never reach
/// the integer arm — no risk of silently truncating a misrouted logp.
///
/// Returns `None` for unsupported Arrow types so the caller can warn-and-skip.
fn column_to_robj(
    cols: &[&dyn Array],
    dtype: &DataType,
    skip: usize,
    n_draws: usize,
    carry_forward: bool,
) -> Option<Robj> {
    let total = cols.len() * n_draws;

    macro_rules! build_doubles {
        ($arr_ty:ty) => {{
            let mut out = Doubles::new(total);
            let dest: &mut [Rfloat] = &mut out;
            let mut i = 0;
            for c in cols {
                let arr = c.as_any().downcast_ref::<$arr_ty>()?;
                for k in 0..n_draws {
                    let row = skip + k;
                    dest[i] = if arr.is_null(row) {
                        Rfloat::na()
                    } else {
                        Rfloat::from(arr.value(row) as f64)
                    };
                    i += 1;
                }
            }
            Some(out.into())
        }};
    }

    macro_rules! build_int_or_double {
        ($arr_ty:ty, $fits:expr) => {{
            let arrs: Vec<&$arr_ty> = cols
                .iter()
                .map(|c| c.as_any().downcast_ref::<$arr_ty>())
                .collect::<Option<Vec<_>>>()?;
            let fits_i32 = arrs.iter().all(|arr| {
                (skip..skip + n_draws).all(|row| arr.is_null(row) || $fits(arr.value(row)))
            });
            if fits_i32 {
                let mut out = Integers::new(total);
                let dest: &mut [Rint] = &mut out;
                let mut i = 0;
                for arr in &arrs {
                    for k in 0..n_draws {
                        let row = skip + k;
                        dest[i] = if arr.is_null(row) {
                            Rint::na()
                        } else {
                            Rint::from(arr.value(row) as i32)
                        };
                        i += 1;
                    }
                }
                Some(out.into())
            } else {
                build_doubles!($arr_ty)
            }
        }};
    }

    match dtype {
        DataType::Boolean => {
            let mut out = Logicals::new(total);
            let dest: &mut [Rbool] = &mut out;
            let mut i = 0;
            for c in cols {
                let arr = c.as_any().downcast_ref::<BooleanArray>()?;
                for k in 0..n_draws {
                    let row = skip + k;
                    dest[i] = if arr.is_null(row) {
                        Rbool::na()
                    } else {
                        Rbool::from(arr.value(row))
                    };
                    i += 1;
                }
            }
            Some(out.into())
        }
        DataType::Utf8 => {
            let mut out = Strings::new_with_na(total);
            let mut i = 0;
            for c in cols {
                let arr = c.as_any().downcast_ref::<StringArray>()?;
                for k in 0..n_draws {
                    let row = skip + k;
                    if !arr.is_null(row) {
                        out.set_elt(i, Rstr::from(arr.value(row)));
                    }
                    i += 1;
                }
            }
            Some(out.into())
        }
        DataType::Float64 => build_doubles!(Float64Array),
        DataType::Float32 => build_doubles!(Float32Array),
        DataType::Int64 => build_int_or_double!(Int64Array, |v: i64| v > i32::MIN as i64
            && v <= i32::MAX as i64),
        DataType::UInt64 => build_int_or_double!(UInt64Array, |v: u64| v <= i32::MAX as u64),
        DataType::Int32 => build_int_or_double!(Int32Array, |v: i32| v > i32::MIN),
        DataType::UInt32 => build_int_or_double!(UInt32Array, |v: u32| v <= i32::MAX as u32),
        DataType::LargeList(inner) if matches!(inner.data_type(), DataType::Float64) => {
            // Uniform-width rows (mass_matrix_inv / _eigvals / _stds — all
            // ndim_unc wide) collapse to a 2-D matrix; mixed widths fall
            // back to a list-of-vectors.
            let arrs: Vec<&LargeListArray> = cols
                .iter()
                .map(|c| c.as_any().downcast_ref::<LargeListArray>())
                .collect::<Option<Vec<_>>>()?;

            let mut uniform_len: Option<usize> = None;
            let mut mixed = false;
            let scan_start = if carry_forward { 0 } else { skip };
            let scan_end = skip + n_draws;
            'outer: for arr in &arrs {
                for row in scan_start..scan_end {
                    if arr.is_null(row) {
                        continue;
                    }
                    let len = arr.value_length(row) as usize;
                    match uniform_len {
                        None => uniform_len = Some(len),
                        Some(prev) if prev == len => {}
                        Some(_) => {
                            mixed = true;
                            break 'outer;
                        }
                    }
                }
            }

            if !mixed {
                if let Some(inner_len) = uniform_len {
                    if inner_len > 0 {
                        let mut out = Doubles::new(total * inner_len);
                        let dest: &mut [Rfloat] = &mut out;
                        for (chain_idx, arr) in arrs.iter().enumerate() {
                            let inner_values = arr
                                .values()
                                .as_any()
                                .downcast_ref::<Float64Array>()?
                                .values();
                            let offsets = arr.value_offsets();
                            // Carry the most recent non-null row's offset
                            // forward; `inner_values` outlives the loop, so
                            // a usize is enough — no per-row Vec clone.
                            let mut last_start: Option<usize> = if carry_forward {
                                (0..skip)
                                    .rev()
                                    .find(|&r| !arr.is_null(r))
                                    .map(|r| offsets[r] as usize)
                            } else {
                                None
                            };
                            for k in 0..n_draws {
                                let row = skip + k;
                                let dest_row = chain_idx * n_draws + k;
                                let src_start = if arr.is_null(row) {
                                    last_start
                                } else {
                                    let s = offsets[row] as usize;
                                    if carry_forward {
                                        last_start = Some(s);
                                    }
                                    Some(s)
                                };
                                match src_start {
                                    Some(start) => {
                                        let src = &inner_values[start..start + inner_len];
                                        for (col, &val) in src.iter().enumerate() {
                                            dest[dest_row + col * total] = Rfloat::from(val);
                                        }
                                    }
                                    None => {
                                        for col in 0..inner_len {
                                            dest[dest_row + col * total] = Rfloat::na();
                                        }
                                    }
                                }
                            }
                        }
                        let mut robj: Robj = out.into();
                        robj.set_attrib("dim", [total as i32, inner_len as i32].into_robj())
                            .ok()?;
                        return Some(robj);
                    }
                }
            }

            let mut out: Vec<Robj> = Vec::with_capacity(total);
            for arr in &arrs {
                let inner_values = arr
                    .values()
                    .as_any()
                    .downcast_ref::<Float64Array>()?
                    .values();
                let offsets = arr.value_offsets();
                for k in 0..n_draws {
                    let row = skip + k;
                    if arr.is_null(row) {
                        out.push(().into_robj());
                        continue;
                    }
                    let start = offsets[row] as usize;
                    let end = offsets[row + 1] as usize;
                    let slice = &inner_values[start..end];
                    if slice.is_empty() {
                        out.push(().into_robj());
                    } else {
                        let robj: Robj = Doubles::from_values(slice.iter().copied()).into();
                        out.push(robj);
                    }
                }
            }
            Some(List::from_values(out).into_robj())
        }
        _ => None,
    }
}

/// Extract diagnostic statistics from sample_stats RecordBatches.
///
/// Schema-driven: iterates `sample_stats.schema().fields()` and dispatches each
/// column through `column_to_robj`. Columns that are entirely null in the
/// requested window are dropped (covers e.g. `mass_matrix_inv` when
/// `store_mass_matrix = false`). Unsupported Arrow types are skipped with a
/// warning rather than failing the whole call.
fn extract_diagnostics(
    results: &[ArrowTrace],
    skip: usize,
    n_draws: usize,
    drop_cols: &[&str],
) -> Result<List> {
    if results.is_empty() {
        return Ok(List::from_values(Vec::<Robj>::new()));
    }

    let schema = results[0].sample_stats.schema();
    let mut names: Vec<String> = Vec::new();
    let mut values: Vec<Robj> = Vec::new();

    for (idx, field) in schema.fields().iter().enumerate() {
        let name = field.name();

        if drop_cols.iter().any(|c| c == name) {
            continue;
        }

        let cols: Vec<&dyn Array> = results
            .iter()
            .map(|t| t.sample_stats.column(idx).as_ref())
            .collect();

        // The mass-matrix snapshots only land on update rows; treat them
        // as piecewise-constant and carry the most recent value forward
        // through the NA gaps. Other columns get the default null-is-NA
        // behaviour. Explicit list, not a prefix match, so a future
        // upstream `mass_matrix_*` column doesn't silently inherit
        // carry-forward semantics it may not warrant.
        let carry_forward = matches!(
            name.as_str(),
            "mass_matrix_inv" | "mass_matrix_eigvals" | "mass_matrix_stds"
        );
        let non_null_start = if carry_forward { 0 } else { skip };
        let any_non_null = cols
            .iter()
            .any(|c| (non_null_start..skip + n_draws).any(|row| row < c.len() && !c.is_null(row)));
        if !any_non_null {
            continue;
        }

        match column_to_robj(&cols, field.data_type(), skip, n_draws, carry_forward) {
            Some(robj) => {
                names.push(name.clone());
                values.push(robj);
            }
            None => {
                rprintln!(
                    "nutpieR: skipping diagnostic '{}' — unsupported Arrow type {:?}",
                    name,
                    field.data_type()
                );
            }
        }
    }

    let pairs: Vec<(&str, Robj)> = names.iter().map(String::as_str).zip(values).collect();
    Ok(List::from_pairs(pairs))
}

/// Open a BridgeStan model and return an `ExternalPtr<BSHandle>` that caches
/// parameter-name metadata. The handle may be used by any of the `bs_*`
/// accessor functions without re-opening the shared library.
/// @noRd
#[extendr]
fn bs_open(lib_path: &str, data_json: &str, seed: i32) -> Robj {
    or_throw((|| -> Result<Robj> {
        let seed_u32 = check_seed(seed)?;
        let handle = model::BSHandle::open(std::path::Path::new(lib_path), data_json, seed_u32)
            .map_err(r_err)?;
        Ok(ExternalPtr::new(handle).into_robj())
    })())
}

/// Block-level parameter names (no transformed parameters / generated
/// quantities), dot-indexed. Length equals `bs_ndim_block()`.
/// @noRd
#[extendr]
fn bs_block_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.block_names.clone()
}

/// Block-level + transformed-parameter names (no generated quantities),
/// dot-indexed. Length equals `param_num(true, false)`. Used by R-side
/// `pars` / `include` resolution to partition names into block / TP / GQ
/// without an extra round-trip into bridgestan.
/// @noRd
#[extendr]
fn bs_block_tp_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.block_tp_names.clone()
}

/// Full constrained parameter names (block + transformed parameters +
/// generated quantities), dot-indexed.
/// @noRd
#[extendr]
fn bs_full_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.full_names.clone()
}

/// Unconstrained parameter names, dot-indexed. Length equals `bs_ndim_unc()`.
/// @noRd
#[extendr]
fn bs_unc_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.unc_names.clone()
}

/// Number of unconstrained parameters.
/// @noRd
#[extendr]
fn bs_ndim_unc(handle: ExternalPtr<model::BSHandle>) -> i32 {
    handle.ndim_unc as i32
}

/// Number of block-level constrained parameters (no TP, no GQ).
/// @noRd
#[extendr]
fn bs_ndim_block(handle: ExternalPtr<model::BSHandle>) -> i32 {
    handle.ndim_block as i32
}

/// Map a flat block-level constrained vector (length `bs_ndim_block()`,
/// BridgeStan column-major / last-index-major order) to the unconstrained
/// space. No JSON parsing.
/// @noRd
#[extendr]
fn bs_param_unconstrain(handle: ExternalPtr<model::BSHandle>, theta: Vec<f64>) -> Vec<f64> {
    or_throw(bs_param_unconstrain_impl(handle, theta))
}

fn bs_param_unconstrain_impl(
    handle: ExternalPtr<model::BSHandle>,
    theta: Vec<f64>,
) -> Result<Vec<f64>> {
    if theta.len() != handle.ndim_block {
        return Err(Error::Other(format!(
            "theta length {} does not match block-level parameter count {}",
            theta.len(),
            handle.ndim_block
        )));
    }
    let mut out = vec![0.0f64; handle.ndim_unc];
    handle
        .model
        .param_unconstrain(&theta, &mut out)
        .map_err(r_err)?;
    Ok(out)
}

/// Map an unconstrained position to the full constrained scale (including
/// transformed parameters and generated quantities) using an already-opened
/// handle.
/// @noRd
#[extendr]
fn bs_param_constrain(
    handle: ExternalPtr<model::BSHandle>,
    theta_unc: Vec<f64>,
    seed: i32,
) -> Vec<f64> {
    or_throw(bs_param_constrain_impl(handle, theta_unc, seed))
}

fn bs_param_constrain_impl(
    handle: ExternalPtr<model::BSHandle>,
    theta_unc: Vec<f64>,
    seed: i32,
) -> Result<Vec<f64>> {
    let seed_u32 = check_seed(seed)?;
    if theta_unc.len() != handle.ndim_unc {
        return Err(Error::Other(format!(
            "theta_unc length {} does not match unconstrained parameter count {}",
            theta_unc.len(),
            handle.ndim_unc
        )));
    }
    let mut out = vec![0.0f64; handle.ndim_full];
    let mut rng = handle.model.new_rng(seed_u32).map_err(r_err)?;
    handle
        .model
        .param_constrain(&theta_unc, true, true, &mut out, Some(&mut rng))
        .map_err(r_err)?;
    Ok(out)
}

/// Map an unconstrained position to the block-level constrained scale only
/// (no transformed parameters, no generated quantities). No RNG is used and
/// no GQ code runs, so this cannot fail on GQ constraint violations — the
/// right primitive for resolving partial-init random fills.
/// @noRd
#[extendr]
fn bs_param_constrain_block(handle: ExternalPtr<model::BSHandle>, theta_unc: Vec<f64>) -> Vec<f64> {
    or_throw(bs_param_constrain_block_impl(handle, theta_unc))
}

fn bs_param_constrain_block_impl(
    handle: ExternalPtr<model::BSHandle>,
    theta_unc: Vec<f64>,
) -> Result<Vec<f64>> {
    if theta_unc.len() != handle.ndim_unc {
        return Err(Error::Other(format!(
            "theta_unc length {} does not match unconstrained parameter count {}",
            theta_unc.len(),
            handle.ndim_unc
        )));
    }
    let mut out = vec![0.0f64; handle.ndim_block];
    let no_rng: Option<&mut bridgestan::Rng<Arc<bridgestan::StanLibrary>>> = None;
    handle
        .model
        .param_constrain(&theta_unc, false, false, &mut out, no_rng)
        .map_err(r_err)?;
    Ok(out)
}

extendr_module! {
    mod nutpieR;
    fn bridgestan_version;
    fn compile_stan_model;
    fn tbb_proxy_live_progress_safe;
    fn ensure_tbb_proxy_patched;
    fn tbb_patch_strings;
    fn sample_stan;
    fn bs_open;
    fn bs_block_names;
    fn bs_block_tp_names;
    fn bs_full_names;
    fn bs_unc_names;
    fn bs_ndim_unc;
    fn bs_ndim_block;
    fn bs_param_unconstrain;
    fn bs_param_constrain;
    fn bs_param_constrain_block;
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    // Guards the verbatim splice: patch_proxy_source() must still find and
    // replace the stock function when it is embedded in surrounding text (#36).
    #[test]
    fn patch_proxy_source_splices_stock_into_surrounding_text() {
        let fixture = format!(
            "// preceding declarations\nstatic void impl_zone_destroy() {{}}\n\n{}\n\n// trailing declarations\n",
            TBB_STOCK_FN
        );
        let patched = patch_proxy_source(&fixture).expect("stock function must be found verbatim");
        assert!(
            patched.contains(TBB_PATCH_MARKER),
            "marker symbol spliced in"
        );
        assert!(
            !patched.contains("malloc_default_zone"),
            "must not probe the virtual default zone (forwards back into us and recurses)"
        );
        assert!(
            patched.contains("malloc_get_all_zones"),
            "registered-zone walk present"
        );
        assert!(patched.contains("// preceding declarations"));
        assert!(patched.contains("// trailing declarations"));
        // The stock body must no longer be present verbatim (it was replaced).
        assert!(!patched.contains(TBB_STOCK_FN));
    }

    #[test]
    fn patch_proxy_source_declines_unknown_source() {
        assert!(patch_proxy_source("nothing familiar here").is_none());
    }

    #[test]
    fn contains_bytes_finds_marker() {
        assert!(contains_bytes(
            b"xxnutpie_tbb_proxy_safe_probeyy",
            TBB_PATCH_MARKER.as_bytes()
        ));
        assert!(!contains_bytes(
            b"unrelated bytes",
            TBB_PATCH_MARKER.as_bytes()
        ));
    }
}
