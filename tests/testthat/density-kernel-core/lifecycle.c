/* Lifecycle instrumentation around the existing Gaussian producer. */
#define nutpier_density_kernel_workspace fixture_workspace
#define nutpier_density_kernel_workspace_destroy fixture_workspace_destroy
#define nutpier_density_kernel_evaluate fixture_evaluate
#define nutpier_density_kernel_destroy fixture_destroy
#include "gaussian.c"
#undef nutpier_density_kernel_workspace
#undef nutpier_density_kernel_workspace_destroy
#undef nutpier_density_kernel_evaluate
#undef nutpier_density_kernel_destroy
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

typedef struct { void *inner; unsigned id; } tracked_workspace;
static atomic_uint next_id = 0;
static atomic_flag log_lock = ATOMIC_FLAG_INIT;
/* Serialize complete records, including open/close; workers never call R. */
static void record(char kind, unsigned id) {
    const char *path = getenv("NUTPIER_LIFECYCLE_LOG");
    char line[64]; FILE *file; int n;
    if (!path) return;
    n = snprintf(line, sizeof(line), "%c %u\n", kind, id);
    while (atomic_flag_test_and_set(&log_lock)) {}
    file = fopen(path, "a");
    if (file) { fwrite(line, 1, (size_t)n, file); fclose(file); }
    atomic_flag_clear(&log_lock);
}
NUTPIER_DENSITY_KERNEL_EXPORT int32_t nutpier_density_kernel_workspace(
    void *b, void **out, char *err, size_t cap) {
    unsigned id = atomic_fetch_add(&next_id, 1) + 1;
    const char *at = getenv("NUTPIER_LIFECYCLE_FAIL_AT");
    tracked_workspace *w; int status;
    *out = NULL; record('A', id);
    if (at && id == (unsigned)strtoul(at, NULL, 10)) {
        record('F', id); return fail(err, cap, "injected workspace factory failure");
    }
    w = calloc(1, sizeof(*w));
    if (!w) return fail(err, cap, "allocation failed");
    w->id = id;
    status = fixture_workspace(b, &w->inner, err, cap);
    if (status) { free(w); return status; }
    *out = w; record('C', id); return 0;
}
NUTPIER_DENSITY_KERNEL_EXPORT void nutpier_density_kernel_workspace_destroy(void *b, void *work) {
    tracked_workspace *w = work;
    fixture_workspace_destroy(b, w->inner);
    record('D', w->id); free(w);
}
NUTPIER_DENSITY_KERNEL_EXPORT void nutpier_density_kernel_destroy(void *b) {
    fixture_destroy(b); record('B', 0);
}
NUTPIER_DENSITY_KERNEL_EXPORT int32_t nutpier_density_kernel_evaluate(
    void *b, void *work, const double *q, size_t n, double *lp, double *g,
    char *err, size_t cap) {
    tracked_workspace *w = work;
    if (((workspace_t *)w->inner)->calls == 0) record('E', w->id);
    if (getenv("NUTPIER_LIFECYCLE_SLOW")) {
#ifdef _WIN32
        Sleep(2);
#else
        struct timespec delay = {0, 2000000}; nanosleep(&delay, NULL);
#endif
    }
    return fixture_evaluate(b, w->inner, q, n, lp, g, err, cap);
}
