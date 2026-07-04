/* PoC for nutpieR issue #36 root cause:
 * tbbmalloc_proxy (TBB 2020.3, also current oneTBB) installs a malloc zone
 * whose size() callback (impl_malloc_usable_size -> __TBB_malloc_safer_msize)
 * is invoked by libmalloc for EVERY free()/malloc_size() to identify the
 * owning zone. For a foreign pointer inside TBB's used-address envelope that
 * is 64-byte aligned, it dereferences ptr-4 (would-be LargeObjectHdr
 * backRefIdx). If ptr sits at the start of a VM region with the preceding
 * page unmapped, that read faults: EXC_BAD_ACCESS at ...fffc.
 *
 * No R, no Stan, no threads needed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/mman.h>
#include <malloc/malloc.h>

int main(int argc, char **argv) {
    const char *proxy = argv[1];

    /* 1. Simulate what R allocated BEFORE any Stan model .so is loaded:
     *    nothing needed yet -- the landmine is constructed below. */

    /* 2. Load the proxy, exactly as dlopen() of a Stan model .so does. */
    void *h = dlopen(proxy, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    printf("proxy loaded: %s\n", proxy);

    /* 3. Grow TBB's usedAddrRange envelope with big allocations through the
     *    now-default TBB zone (stand-in for a large Stan model sampling). */
    enum { NBLK = 64 };
    void *blk[2 * NBLK];
    for (int i = 0; i < NBLK; i++) {
        blk[i] = malloc(8u << 20);
        memset(blk[i], 1, 8u << 20);
    }
    uintptr_t lo = (uintptr_t)blk[0], hi = lo;
    for (int i = 0; i < NBLK; i++) {
        uintptr_t p = (uintptr_t)blk[i];
        if (p < lo) lo = p;
        if (p > hi) hi = p;
    }

    /* 4. Construct the landmine: a page-aligned pointer at the start of a VM
     *    region with the preceding page unmapped. This is what a pre-proxy
     *    MALLOC_LARGE block (e.g. a large R vector allocated before the model
     *    was compiled) looks like. Place it above TBB's current high mark,
     *    then extend the envelope past it with more TBB allocations. */
    size_t pg = (size_t)getpagesize();
    void *hint = (void *)(((hi + (64u << 20)) + pg - 1) & ~(pg - 1));
    void *m = mmap(hint, 2 * pg, PROT_READ | PROT_WRITE,
                   MAP_ANON | MAP_PRIVATE, -1, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 2; }
    munmap(m, pg);
    char *landmine = (char *)m + pg;

    for (int i = NBLK; i < 2 * NBLK; i++) {
        blk[i] = malloc(8u << 20);
        memset(blk[i], 1, 8u << 20);
        uintptr_t p = (uintptr_t)blk[i];
        if (p < lo) lo = p;
        if (p > hi) hi = p;
    }
    printf("tbb blocks span   %p .. %p\n", (void *)lo, (void *)hi);
    printf("landmine          %p (in span: %s)\n", (void *)landmine,
           ((uintptr_t)landmine > lo && (uintptr_t)landmine < hi) ? "yes" : "no");

    /* 5. What free() does first: libmalloc asks each registered zone
     *    "is this yours?" via zone->size(). malloc_size() runs the same
     *    probe. TBB's zone is asked before the system zone. */
    printf("probing malloc_size(landmine)... expect EXC_BAD_ACCESS at %p\n",
           (void *)(landmine - 4));
    fflush(stdout);
    size_t sz = malloc_size(landmine);
    printf("SURVIVED: malloc_size = %zu\n", sz);
    return 0;
}
