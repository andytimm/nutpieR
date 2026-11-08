# nutpieR patch to nuts-rs 0.19.1

Source: the published crates.io nuts-rs 0.19.1 package (https://crates.io/crates/nuts-rs/0.19.1).

Only `ChainJob::run` in `src/sampler.rs` is changed. Catch worker panics around
sampling and local teardown, convert the panic payload to a chain-contextual
error, and always notify the result channel. Without this catch, a worker panic
with queued chains can leave the host polling forever: queued jobs retain result
senders while the controller waits for commands and its worker scope cannot join.
The existing error path then aborts and joins the controller and workers.

`tests/queued_panic.rs` reproduces this with three chains and one worker. Run it
under an external subprocess timeout; an in-process test timeout cannot stop a
blocked controller. No join-on-drop behavior or other sampler changes are added.
