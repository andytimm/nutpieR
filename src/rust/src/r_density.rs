//! Log-density backed by user-supplied R closures (issue #26).
//!
//! Lets R users sample an arbitrary log-posterior + gradient — e.g. a
//! preconditioned RTMB/TMB model — with nuts-rs, without going through Stan.
//!
//! Unlike [`crate::model::StanModel`], this density holds non-`Send` extendr
//! `Function`s, so it CANNOT be driven by the parallel `Sampler` (rayon spawns
//! `logp` onto worker threads, and R is single-threaded). Instead the caller
//! drives a single `NutsChain` synchronously on the main R thread via
//! `Settings::new_chain` + `Chain::draw` (see `sample_r_density` in `lib.rs`).
//! That is sound because `CpuLogpFunc`/`Math` only require `Send`/`Sync` on
//! their associated *error* types, not on `self`.

use extendr_api::prelude::*;
use nuts_rs::{CpuLogpFunc, CpuMathError, HasDims, LogpError};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

/// Errors raised while evaluating the R log-density / gradient callbacks.
///
/// Bad numeric *values* (non-finite logp or gradient) are recoverable: nuts-rs
/// treats them as divergences and keeps sampling, matching how Stan handles
/// `-Inf` log densities. A failed or malformed R call is non-recoverable —
/// that is almost always a bug in the user's function, so we abort and surface
/// the R error rather than silently turning it into divergences.
#[derive(Debug, thiserror::Error)]
pub enum RLogpError {
    #[error("R callback raised an error: {0}")]
    Call(String),
    #[error("R callback returned an unexpected value: {0}")]
    BadReturn(String),
    #[error("non-finite log density: {0}")]
    BadLogp(f64),
    #[error("non-finite gradient element at index {0}")]
    BadGradient(usize),
}

impl LogpError for RLogpError {
    fn is_recoverable(&self) -> bool {
        matches!(self, RLogpError::BadLogp(_) | RLogpError::BadGradient(_))
    }
}

/// Per-chain log-density evaluator wrapping R closures.
///
/// `logp_fn(y) -> double` returns the log-posterior (log-density convention,
/// i.e. higher = more probable — already sign-flipped from TMB's negative LP).
/// `grad_fn(y) -> double[ndim]` returns its gradient. The faster path is a
/// combined `value_grad_fn(y) -> c(logp, gradient)`, called once per leapfrog
/// step so expensive transforms can be shared between value and gradient.
pub struct RDensity {
    logp_fn: Option<Function>,
    grad_fn: Option<Function>,
    value_grad_fn: Option<Function>,
    ndim: usize,
    stats: Rc<RefCell<CallbackStats>>,
}

#[derive(Debug, Default)]
pub struct CallbackStats {
    pub logp_evals: usize,
    pub r_calls: usize,
    pub elapsed: Duration,
}

impl RDensity {
    pub fn new(
        logp_fn: Option<Function>,
        grad_fn: Option<Function>,
        value_grad_fn: Option<Function>,
        ndim: usize,
        stats: Rc<RefCell<CallbackStats>>,
    ) -> Self {
        RDensity {
            logp_fn,
            grad_fn,
            value_grad_fn,
            ndim,
            stats,
        }
    }

    /// Call an R closure with the position vector and return its result.
    fn call(f: &Function, position: &[f64]) -> std::result::Result<Robj, RLogpError> {
        // `Robj::from(&[f64])` builds the R vector straight from the slice — no
        // intermediate `Vec` alloc. This runs twice per leapfrog step.
        f.call(Pairlist::from_pairs([("", Robj::from(position))]))
            .map_err(|e| RLogpError::Call(e.to_string()))
    }

    fn logp_inner(
        &mut self,
        position: &[f64],
        gradient: &mut [f64],
        r_calls: &mut usize,
    ) -> std::result::Result<f64, RLogpError> {
        if let Some(ref value_grad_fn) = self.value_grad_fn {
            *r_calls += 1;
            let vg_robj = Self::call(value_grad_fn, position)?;
            let vg = vg_robj.as_real_slice().ok_or_else(|| {
                RLogpError::BadReturn("value_grad callback did not return a numeric vector".into())
            })?;
            if vg.len() != self.ndim + 1 {
                return Err(RLogpError::BadReturn(format!(
                    "value_grad callback returned length {}, expected {} (logp plus {} gradient entries)",
                    vg.len(),
                    self.ndim + 1,
                    self.ndim
                )));
            }
            let lp = vg[0];
            if !lp.is_finite() {
                return Err(RLogpError::BadLogp(lp));
            }
            for (i, (g, &src)) in gradient.iter_mut().zip(vg[1..].iter()).enumerate() {
                if !src.is_finite() {
                    return Err(RLogpError::BadGradient(i));
                }
                *g = src;
            }
            return Ok(lp);
        }

        // Gradient first: with RTMB/TMB the forward pass is cached at the last
        // evaluation point, so calling the value at the same `position` right
        // after reuses that tape instead of recomputing it.
        let grad_fn = self
            .grad_fn
            .as_ref()
            .ok_or_else(|| RLogpError::BadReturn("missing gradient callback".into()))?;
        let logp_fn = self
            .logp_fn
            .as_ref()
            .ok_or_else(|| RLogpError::BadReturn("missing log-density callback".into()))?;
        *r_calls += 1;
        let grad_robj = Self::call(grad_fn, position)?;
        // `as_real_slice` borrows the R vector's data (no copy); `grad_robj`
        // owns it until end of scope, so the borrow stays valid.
        let grad = grad_robj.as_real_slice().ok_or_else(|| {
            RLogpError::BadReturn("gradient callback did not return a numeric vector".into())
        })?;
        if grad.len() != self.ndim {
            return Err(RLogpError::BadReturn(format!(
                "gradient callback returned length {}, expected {}",
                grad.len(),
                self.ndim
            )));
        }
        for (i, (g, &src)) in gradient.iter_mut().zip(grad.iter()).enumerate() {
            if !src.is_finite() {
                return Err(RLogpError::BadGradient(i));
            }
            *g = src;
        }
        *r_calls += 1;
        let lp_robj = Self::call(logp_fn, position)?;
        let lp = lp_robj.as_real().ok_or_else(|| {
            RLogpError::BadReturn("log-density callback did not return a numeric scalar".into())
        })?;
        if !lp.is_finite() {
            return Err(RLogpError::BadLogp(lp));
        }
        Ok(lp)
    }
}

impl HasDims for RDensity {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("unc".to_string(), self.ndim as u64)])
    }
}

impl CpuLogpFunc for RDensity {
    type LogpError = RLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.ndim
    }

    fn logp(
        &mut self,
        position: &[f64],
        gradient: &mut [f64],
    ) -> std::result::Result<f64, Self::LogpError> {
        let start = Instant::now();
        let mut r_calls = 0usize;
        let result = self.logp_inner(position, gradient, &mut r_calls);
        let mut stats = self.stats.borrow_mut();
        stats.logp_evals += 1;
        stats.r_calls += r_calls;
        stats.elapsed += start.elapsed();
        result
    }

    fn expand_vector<R>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> std::result::Result<Self::ExpandedVector, CpuMathError>
    where
        R: rand::Rng + ?Sized,
    {
        // Identity: the sampled (preconditioned) position IS the reported draw.
        // Any back-transform y -> x or generated quantities is applied R-side
        // via `expand_fn`, once per kept draw — never in this hot loop.
        Ok(array.to_vec())
    }
}
