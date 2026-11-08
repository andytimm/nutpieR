//! One host owner for the nuts-rs 0.19.1 threaded sampler lifecycle.
use super::{model, r_err};
use extendr_api::error::Result;
use extendr_api::prelude::*;
use nuts_rs::{
    ArrowConfig, ArrowTrace, ChainProgress, ProgressCallback, Sampler, SamplerWaitResult, Settings,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::time::Duration;

extern "C" {
    static mut R_interrupts_pending: std::os::raw::c_int;
    static mut R_interrupts_suspended: std::os::raw::c_int;
    fn R_ProcessEvents();
    fn R_ToplevelExec(
        fun: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
        data: *mut std::ffi::c_void,
    ) -> std::os::raw::c_int;
}

// Keep the sampler owner outside every protected R frame. R errors cannot
// longjmp over its teardown. R may skip temporary callback allocations, not
// the retained owner. Rust panics never cross the C callback boundary.
fn protected_r<F: FnOnce() -> T, T>(f: F) -> Result<T> {
    struct Call<F, T> {
        function: Option<F>,
        result: Option<std::thread::Result<T>>,
    }
    unsafe extern "C" fn invoke<F: FnOnce() -> T, T>(data: *mut std::ffi::c_void) {
        let call = &mut *data.cast::<Call<F, T>>();
        call.result = Some(catch_unwind(AssertUnwindSafe(|| {
            (call.function.take().unwrap())()
        })));
    }
    let mut call = Call {
        function: Some(f),
        result: None,
    };
    let ok = unsafe { R_ToplevelExec(Some(invoke::<F, T>), (&mut call as *mut Call<F, T>).cast()) };
    if ok == 0 {
        return Err(r_err("Error in sampling host monitor."));
    }
    match call.result.unwrap() {
        Ok(value) => Ok(value),
        Err(panic) => Err(r_err(format!(
            "Sampler monitor panicked: {}",
            panic_text(panic)
        ))),
    }
}

fn interrupt_pending() -> bool {
    unsafe {
        let pending = R_interrupts_pending != 0;
        R_interrupts_pending = 0;
        pending
    }
}

fn pump_r_events() -> Result<()> {
    unsafe {
        let previous = R_interrupts_suspended;
        R_interrupts_suspended = 1;
        let result = protected_r(|| R_ProcessEvents());
        R_interrupts_suspended = previous;
        result
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
    latest_num_gradients: usize,
    total_num_gradients: usize,
    step_size: f64,
    runtime: Duration,
    divergent_draws: Vec<usize>,
    walnuts: bool,
}

/// Extract a caught Rust panic message for the R-facing error.
fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).into()
    } else {
        "non-string Rust panic".into()
    }
}

/// Wait transfers ownership only into the reviewed 0.19.1 threaded driver.
/// Its terminal paths call abort(), which joins before returning or rethrowing
/// a controller panic. This is source-specific, not a generic unwind guarantee.
/// The vendored ChainJob catch also makes queued worker panics terminal.
pub(super) fn run_sampler<S: Settings>(
    stan_model: model::StanModel,
    settings: S,
    num_cores: i32,
    save_warmup: bool,
    mut progress_cb: Option<Function>,
    walnuts: bool,
) -> Result<Vec<ArrowTrace>> {
    let native_error = stan_model.run_state.clone();
    let chains = settings.num_chains();
    let rows = settings.hint_num_draws()
        + if save_warmup {
            settings.hint_num_tune()
        } else {
            0
        };
    let progress = Arc::new(Mutex::new(Vec::<ChainState>::new()));
    let snapshot = progress.clone();
    let callback = progress_cb.as_ref().map(|_| ProgressCallback {
        callback: Box::new(move |_, values: Box<[ChainProgress]>| {
            *snapshot.lock().unwrap_or_else(|e| e.into_inner()) = values
                .iter()
                .enumerate()
                .map(|(i, c)| ChainState {
                    chain: i + 1,
                    walnuts,
                    finished_draws: c.finished_draws,
                    total_draws: c.total_draws,
                    divergences: c.divergences,
                    tuning: c.tuning,
                    started: c.started,
                    latest_num_steps: c.latest_num_steps,
                    total_num_steps: c.total_num_steps,
                    latest_num_gradients: c.latest_num_gradients,
                    total_num_gradients: c.total_num_gradients,
                    step_size: c.step_size,
                    runtime: c.runtime,
                    divergent_draws: c.divergent_draws.clone(),
                })
                .collect();
        }),
        rate: Duration::from_millis(100),
    });
    let mut config = ArrowConfig::default();
    config.store_warmup = save_warmup;
    let created = catch_unwind(AssertUnwindSafe(|| {
        Sampler::new(
            Arc::new(stan_model),
            settings,
            config,
            num_cores as usize,
            callback,
        )
    }));
    let sampler = match created {
        Ok(Ok(sampler)) => sampler,
        Ok(Err(error)) => return Err(r_err(error)),
        Err(panic) => {
            return Err(r_err(format!(
                "Sampler startup panicked: {}",
                panic_text(panic)
            )))
        }
    };
    let mut owner = Some(sampler);
    let outcome = loop {
        let sampler = owner.take().unwrap();
        match catch_unwind(AssertUnwindSafe(|| {
            sampler.wait_timeout(Duration::from_millis(200))
        })) {
            Ok(SamplerWaitResult::Timeout(sampler)) => owner = Some(sampler),
            Ok(SamplerWaitResult::Trace(traces)) => break Ok(traces),
            Ok(SamplerWaitResult::Err(error, _)) => break Err(r_err(error)),
            Err(panic) => break Err(r_err(format!("Sampler panicked: {}", panic_text(panic)))),
        }
        // Owner has been restored before any host code runs.
        let monitored = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
            if interrupt_pending() {
                return Err(r_err("Sampling interrupted."));
            }
            pump_r_events()?;
            if interrupt_pending() {
                return Err(r_err("Sampling interrupted."));
            }
            report_progress(&progress, &mut progress_cb)
        }));
        let failure = match monitored {
            Ok(Ok(())) => continue,
            Ok(Err(error)) => error,
            Err(panic) => r_err(format!("Sampler monitor panicked: {}", panic_text(panic))),
        };
        // abort joins before rethrowing a controller panic; preserve the host
        // failure unless the kernel retained a more actionable native error.
        let _joined = catch_unwind(AssertUnwindSafe(|| owner.take().unwrap().abort()));
        break Err(failure);
    };
    if let Some(error) = native_error.and_then(|state| state.error()) {
        return Err(r_err(error));
    }
    let traces = outcome?;
    if traces.len() != chains
        || traces.iter().any(|trace| {
            trace.posterior.num_rows() != rows || trace.sample_stats.num_rows() != rows
        })
    {
        let actual: Vec<_> = traces
            .iter()
            .map(|trace| (trace.posterior.num_rows(), trace.sample_stats.num_rows()))
            .collect();
        return Err(r_err(format!("Sampling stopped before all draws completed; run discarded. Expected {chains} chains with {rows} rows; got {actual:?}.")));
    }
    // Terminal results are joined before the final R callback.
    report_progress(&progress, &mut progress_cb)?;
    Ok(traces)
}

fn report_progress(
    progress: &Mutex<Vec<ChainState>>,
    callback: &mut Option<Function>,
) -> Result<()> {
    let Some(cb) = callback.as_ref() else {
        return Ok(());
    };
    let snapshot = progress.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if snapshot.is_empty() {
        return Ok(());
    }
    // Defer automatic R interrupt processing until the callback returns so a
    // Ctrl-C cannot be consumed as an ordinary callback error. Explicit R
    // interrupt conditions are tagged by the public callback guard.
    let previous = unsafe { R_interrupts_suspended };
    unsafe {
        R_interrupts_suspended = 1;
    }
    let called = protected_r(|| {
        let arguments =
            Pairlist::from_pairs([("", Robj::from(build_progress_snapshot(&snapshot)))]);
        cb.call(arguments)
    });
    unsafe {
        R_interrupts_suspended = previous;
    }
    if interrupt_pending() {
        return Err(r_err("Sampling interrupted."));
    }
    let failure = match called {
        Ok(Ok(value)) if value.inherits("nutpie_sampling_interrupt") => {
            return Err(r_err("Sampling interrupted."));
        }
        Ok(Ok(_)) => return Ok(()),
        Ok(Err(error)) | Err(error) => error,
    };
    *callback = None;
    protected_r(|| {
        rprintln!(
            "nutpieR: progress callback failed ({}); disabling further callbacks for this run.",
            failure
        );
    })
}

/// Build a `List` of per-chain named lists for the R progress callback.
/// Field set must match the R-side progress helpers in `R/progress.R`.
fn build_progress_snapshot(state: &[ChainState]) -> List {
    let entries: Vec<Robj> = state
        .iter()
        .map(|c| {
            list!(
                chain = c.chain as i32,
                walnuts = c.walnuts,
                finished_draws = c.finished_draws as i32,
                total_draws = c.total_draws as i32,
                divergences = c.divergences as i32,
                tuning = c.tuning,
                started = c.started,
                latest_num_steps = c.latest_num_steps as i32,
                total_num_steps = c.total_num_steps as f64,
                latest_num_gradients = c.latest_num_gradients as f64,
                total_num_gradients = c.total_num_gradients as f64,
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
