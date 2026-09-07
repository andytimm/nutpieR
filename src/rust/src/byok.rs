//! Trusted in-process kernel ABI. No R API is called by kernel owners or workspaces.
use libloading::Library;
use std::ffi::{c_char, c_void};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

type Bind = unsafe extern "C" fn(
    *const c_char,
    usize,
    usize,
    *const c_char,
    usize,
    *mut *mut c_void,
    *mut c_char,
    usize,
) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Create = unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut c_char, usize) -> i32;
type DropWorkspace = unsafe extern "C" fn(*mut c_void, *mut c_void);
type Evaluate = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const f64,
    usize,
    *mut f64,
    *mut f64,
    *mut c_char,
    usize,
) -> i32;

pub struct BoundKernel {
    _library: Library,
    bound: *mut c_void,
    destroy: Destroy,
    create: Create,
    drop_workspace: DropWorkspace,
    evaluate: Evaluate,
    pub ndim: usize,
}
// ABI requires shared immutable bound data, concurrent factories/evaluation,
// and permits destruction on another thread. Library lives through every owner.
unsafe impl Send for BoundKernel {}
unsafe impl Sync for BoundKernel {}
impl Drop for BoundKernel {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.bound) }
    }
}
#[derive(Debug, Clone, thiserror::Error)]
#[error("Kernel status {status}: {message}")]
pub struct KernelError {
    pub status: i32,
    pub message: String,
}
impl nuts_rs::LogpError for KernelError {
    fn is_recoverable(&self) -> bool {
        self.status == 1
    }
}
fn error(status: i32, bytes: &[u8]) -> KernelError {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let text = String::from_utf8_lossy(&bytes[..end]);
    KernelError {
        status: if status == 1 { 1 } else { 2 },
        message: if text.is_empty() {
            format!("native call returned status {status} without a message")
        } else {
            format!("{text} (native status {status})")
        },
    }
}
impl BoundKernel {
    pub fn bind(
        path: &str,
        json: &str,
        ndim: usize,
        names: &[String],
    ) -> anyhow::Result<Arc<Self>> {
        anyhow::ensure!(
            names.len() == ndim,
            "Reference layout length differs from dimension"
        );
        let layout = names.join("\n");
        // Loading executes trusted native initializers. Resolve every symbol and
        // version before allowing a constructor to publish an owned allocation.
        unsafe {
            let library = Library::new(path)?;
            let version =
                *library.get::<unsafe extern "C" fn() -> u32>(b"nutpier_kernel_abi_version\0")?;
            anyhow::ensure!(
                version() == 1,
                "Unsupported kernel ABI version (expected 1)"
            );
            let bind = *library.get::<Bind>(b"nutpier_kernel_bind\0")?;
            let destroy = *library.get::<Destroy>(b"nutpier_kernel_destroy\0")?;
            let create = *library.get::<Create>(b"nutpier_kernel_workspace\0")?;
            let drop_workspace =
                *library.get::<DropWorkspace>(b"nutpier_kernel_workspace_destroy\0")?;
            let evaluate = *library.get::<Evaluate>(b"nutpier_kernel_evaluate\0")?;
            let mut bound = std::ptr::null_mut();
            let mut message = [0u8; 1024];
            let status = bind(
                json.as_ptr().cast(),
                json.len(),
                ndim,
                layout.as_ptr().cast(),
                layout.len(),
                &mut bound,
                message.as_mut_ptr().cast(),
                message.len(),
            );
            if status != 0 {
                return Err(error(status, &message).into());
            }
            Ok(Arc::new(Self {
                _library: library,
                bound,
                destroy,
                create,
                drop_workspace,
                evaluate,
                ndim,
            }))
        }
    }
    pub fn workspace(self: &Arc<Self>) -> anyhow::Result<Workspace> {
        let mut pointer = std::ptr::null_mut();
        let mut message = [0u8; 1024];
        let status = unsafe {
            (self.create)(
                self.bound,
                &mut pointer,
                message.as_mut_ptr().cast(),
                message.len(),
            )
        };
        if status != 0 {
            return Err(error(status, &message).into());
        }
        Ok(Workspace {
            kernel: self.clone(),
            pointer,
            _same_thread: PhantomData,
        })
    }
}
// Deliberately !Send and !Sync. The checker and nuts-rs factory keep this value
// on its creation thread, including destruction during Rust unwinding.
pub struct Workspace {
    kernel: Arc<BoundKernel>,
    pointer: *mut c_void,
    _same_thread: PhantomData<Rc<()>>,
}
impl Drop for Workspace {
    fn drop(&mut self) {
        unsafe { (self.kernel.drop_workspace)(self.kernel.bound, self.pointer) }
    }
}
impl Workspace {
    pub fn evaluate(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, KernelError> {
        let fail = |message: &str| KernelError {
            status: 2,
            message: message.into(),
        };
        if position.len() != self.kernel.ndim || gradient.len() != self.kernel.ndim {
            return Err(fail(
                "position/gradient length differs from bound dimension",
            ));
        }
        if !position.iter().all(|x| x.is_finite()) {
            return Err(KernelError {
                status: 1,
                message: "Nonfinite proposal rejected before kernel evaluation".into(),
            });
        }
        // Poisoning catches partial writes, including stale finite scratch output.
        gradient.fill(f64::NAN);
        let mut logp = f64::NAN;
        let mut message = [0u8; 1024];
        let status = unsafe {
            (self.kernel.evaluate)(
                self.kernel.bound,
                self.pointer,
                position.as_ptr(),
                position.len(),
                &mut logp,
                gradient.as_mut_ptr(),
                message.as_mut_ptr().cast(),
                message.len(),
            )
        };
        if status != 0 {
            return Err(error(status, &message));
        }
        if !logp.is_finite() || !gradient.iter().all(|x| x.is_finite()) {
            return Err(fail(
                "successful evaluation returned nonfinite or unwritten output",
            ));
        }
        Ok(logp)
    }
}

/// Session-local external pointer payload. No R objects inside native ownership.
pub struct KernelHandle(pub Arc<BoundKernel>);

/// Per-run native notification, never persisted in a binding. A completion
/// notification is only a wakeup: the host still joins through Sampler::abort.
#[derive(Default)]
pub struct RunState {
    // Healthy evaluations need only an atomic read, not a shared mutex.
    error_present: std::sync::atomic::AtomicBool,
    inner: std::sync::Mutex<(bool, Option<String>, usize)>,
    wake: std::sync::Condvar,
}
impl RunState {
    pub fn fail(&self, error: impl std::fmt::Display) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.1.is_none() {
            inner.1 = Some(error.to_string());
        }
        self.error_present
            .store(true, std::sync::atomic::Ordering::Release);
        self.wake.notify_all();
    }
    pub fn complete(&self) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).0 = true;
        self.wake.notify_all();
    }
    pub fn wait(&self, duration: std::time::Duration, chains: usize) -> bool {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (inner, _) = self
            .wake
            .wait_timeout_while(inner, duration, |v| !v.0 && v.1.is_none() && v.2 < chains)
            .unwrap_or_else(|e| e.into_inner());
        inner.0 || inner.1.is_some() || inner.2 >= chains
    }
    pub fn error(&self) -> Option<String> {
        if !self
            .error_present
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return None;
        }
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .1
            .clone()
    }
}

/// Created at worker factory entry; moved into Math on success. Schema-only
/// factory gets no guard. Counts terminal factory/Math lifetimes, not chains IDs.
pub struct WorkerGuard(pub Arc<RunState>);
impl Drop for WorkerGuard {
    fn drop(&mut self) {
        self.0.inner.lock().unwrap_or_else(|e| e.into_inner()).2 += 1;
        self.0.wake.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn error_buffers_are_bounded_and_unknown_status_is_fatal() {
        let error = error(97, &[b'x'; 1024]);
        assert_eq!(error.status, 2);
        assert!(error.message.starts_with(&"x".repeat(1024)));
        assert_eq!(super::error(1, b"domain\0ignored").status, 1);
        assert!(super::error(2, b"\0").message.contains("without a message"));
    }
    #[test]
    fn concurrent_failure_publishes_one_owned_error_and_wakes_monitor() {
        let state = Arc::new(RunState::default());
        let barrier = Arc::new(std::sync::Barrier::new(3));
        std::thread::scope(|scope| {
            for message in ["first-a", "first-b"] {
                let state = state.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    barrier.wait();
                    state.fail(message);
                });
            }
            barrier.wait();
            assert!(state.wait(std::time::Duration::from_secs(2), usize::MAX));
            let winner = state.error().expect("failure published before wakeup");
            assert!(matches!(winner.as_str(), "first-a" | "first-b"));
            // The first string cannot be replaced by the second writer.
            state.fail("later");
            assert_eq!(state.error().as_deref(), Some(winner.as_str()));
        });
        assert!(RunState::default().error().is_none());
    }
    #[test]
    fn completion_is_a_wakeup_and_first_error_is_owned() {
        let state = Arc::new(RunState::default());
        assert!(state.error().is_none());
        assert!(!state.wait(std::time::Duration::ZERO, 1));
        drop(WorkerGuard(state.clone()));
        assert!(state.wait(std::time::Duration::ZERO, 1));
        state.fail("first");
        state.fail("second");
        assert_eq!(state.error().as_deref(), Some("first"));
    }
}
