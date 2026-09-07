#ifndef NUTPIER_KERNEL_V1_H
#define NUTPIER_KERNEL_V1_H
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define NUTPIER_KERNEL_EXPORT __declspec(dllexport)
#else
#define NUTPIER_KERNEL_EXPORT __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental trusted in-process ABI. All functions use the C calling convention.
 * Inputs are borrowed for the call only. Lengths count bytes/elements, not NUL.
 * Status: 0 success, 1 domain rejection (evaluate only), 2 fatal; others fatal.
 * Error text is UTF-8 in host buffer; NUL terminate when capacity permits.
 * Constructors clean partial allocations on failure and publish no handle.
 * Bound data is immutable, concurrently readable; factories may run concurrently.
 * Workspaces are created, evaluated and destroyed on one thread. NULL is valid.
 * Bound/library destruction may run on another thread. Destructors never throw.
 * No exceptions or panics may cross this ABI. All handles survive failed calls.
 * Successful evaluation fills finite logp and every gradient element. Buffers do
 * not alias; failed outputs are ignored. Recoverable errors leave scratch reusable.
 * Density convention: BridgeStan propto=true, jacobian=true, exact ordered
 * unconstrained coordinates. Layout is newline-separated BridgeStan unc names,
 * without a trailing newline. Bind must reject dimension/layout mismatches.
 * All doubles are IEEE-754 binary64. Successful bound and workspace handles may
 * be NULL; their destructors must accept these successful NULL handles.
 * Callbacks must not call R or retain borrowed buffers for background work.
 * Error capacity is bytes; write at most capacity bytes (none if zero). Host
 * scans at most capacity bytes, decodes lossily and supplies a fallback when
 * empty. Long messages may be truncated; no trailing NUL is required by host.
 * JSON follows nutpie_attach_kernel documentation (jsonlite auto_unbox=TRUE,
 * digits=NA for lists; JSON strings/files unchanged; absent data becomes {}).
 * Parse values, not byte formatting.
 * Fixed-data kernels must verify meaningful supplied values against embedded data.
 */
NUTPIER_KERNEL_EXPORT uint32_t nutpier_kernel_abi_version(void);
NUTPIER_KERNEL_EXPORT int32_t nutpier_kernel_bind(const char *json, size_t json_len,
    size_t ndim, const char *layout, size_t layout_len, void **bound,
    char *error, size_t error_capacity);
NUTPIER_KERNEL_EXPORT void nutpier_kernel_destroy(void *bound);
NUTPIER_KERNEL_EXPORT int32_t nutpier_kernel_workspace(void *bound, void **workspace,
    char *error, size_t error_capacity);
NUTPIER_KERNEL_EXPORT void nutpier_kernel_workspace_destroy(void *bound, void *workspace);
NUTPIER_KERNEL_EXPORT int32_t nutpier_kernel_evaluate(void *bound, void *workspace,
    const double *position, size_t ndim, double *logp, double *gradient,
    char *error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
