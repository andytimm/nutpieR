use nuts_rs::{
    CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HashMapConfig, InitPositionError,
    LogpError, Model, Sampler, SamplerWaitResult,
};
use rand::Rng;
use std::{
    sync::Arc,
    thread::{self, ThreadId},
    time::Duration,
};
#[derive(Debug, thiserror::Error)]
#[error("logp error")]
struct E;
impl LogpError for E {
    fn is_recoverable(&self) -> bool {
        false
    }
}
struct Density {
    created_on: ThreadId,
}
impl Drop for Density {
    fn drop(&mut self) {
        eprintln!(
            "DENSITY_DROP same_thread={} thread={:?}",
            self.created_on == thread::current().id(),
            thread::current().id()
        );
    }
}
impl nuts_storable::HasDims for Density {
    fn dim_sizes(&self) -> std::collections::HashMap<String, u64> {
        [("unconstrained_parameter".into(), 1), ("dim".into(), 1)].into()
    }
}
impl CpuLogpFunc for Density {
    type LogpError = E;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;
    fn dim(&self) -> usize {
        1
    }
    fn logp(&mut self, x: &[f64], g: &mut [f64]) -> Result<f64, E> {
        g[0] = -x[0];
        Ok(-x[0] * x[0] / 2.)
    }
    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        _: &mut R,
        x: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        Ok(x.to_vec())
    }
}
struct M;
impl Model for M {
    type Math = CpuMath<Density>;
    fn math<R: Rng + ?Sized>(self: Arc<Self>, _: &mut R) -> anyhow::Result<Self::Math> {
        eprintln!("DENSITY_NEW thread={:?}", thread::current().id());
        Ok(CpuMath::new(Density {
            created_on: thread::current().id(),
        }))
    }
    fn init_position<R: Rng + ?Sized>(
        &self,
        _: &mut R,
        id: u64,
        _: &mut [f64],
    ) -> Result<(), InitPositionError> {
        eprintln!("INIT chain={id}; injecting panic");
        panic!("injected worker panic");
    }
}
#[test]
fn queued_worker_panic_returns_an_error() {
    let mut settings = DiagNutsSettings::default();
    settings.num_chains = 3;
    settings.num_tune = 10;
    settings.num_draws = 10;
    let mut sampler =
        Sampler::new(Arc::new(M), settings, HashMapConfig::default(), 1, None).unwrap();
    loop {
        match sampler.wait_timeout(Duration::from_millis(100)) {
            SamplerWaitResult::Timeout(owner) => sampler = owner,
            SamplerWaitResult::Err(error, _) => {
                let message = format!("{error:#}");
                assert!(
                    message.contains("Chain 0 panicked: injected worker panic"),
                    "{message}"
                );
                break;
            }
            SamplerWaitResult::Trace(_) => panic!("panic must not produce successful traces"),
        }
    }
}
