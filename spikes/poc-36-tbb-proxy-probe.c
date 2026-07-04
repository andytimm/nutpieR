/* Realistic PoC for nutpieR #36, corrected: large blocks allocated with plain
 * malloc() BEFORE the proxy loads (exactly like R data / large R vectors that
 * exist before the Stan model .so is dlopen'd). After the proxy swaps the
 * default zone and TBB's envelope grows, freeing one of those pre-proxy
 * blocks that sits at a VM region start with an unmapped page below it makes
 * TBB's ownership probe read region_start-4 -> EXC_BAD_ACCESS.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <malloc/malloc.h>
#include <mach/mach.h>

static int page_below_unmapped(void *p, size_t pg) {
    return msync((char *)p - pg, pg, MS_ASYNC) != 0 && errno == ENOMEM;
}

int main(int argc, char **argv) {
    const char *proxy = argv[1];
    size_t pg = (size_t)getpagesize();

    /* Concrete system zone (registered zone 0), not the virtual wrapper. */
    vm_address_t *zaddr; unsigned zcnt = 0;
    malloc_get_all_zones(mach_task_self(), NULL, &zaddr, &zcnt);
    malloc_zone_t *szone = (malloc_zone_t *)zaddr[0];

    /* 1. Pre-proxy allocations, like R data loaded before model compile. */
    enum { NSZ = 256 };
    const size_t SZBLK = 16u << 20;
    char *sz[NSZ];
    for (int i = 0; i < NSZ; i++) { sz[i] = malloc(SZBLK); sz[i][0] = 2; }
    printf("szone owns sz[0]: %s\n",
           szone->size(szone, sz[0]) > 0 ? "yes" : "NO (bug in poc)");

    /* 2. Proxy loads (= dlopen of the Stan model .so). */
    void *h = dlopen(proxy, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }

    /* 3. TBB envelope grows (= large-model sampling) until it spans past
     * every pre-proxy block. */
    uintptr_t szlo = (uintptr_t)sz[0], szhi = szlo;
    for (int i = 0; i < NSZ; i++) {
        uintptr_t p = (uintptr_t)sz[i];
        if (p < szlo) szlo = p;
        if (p > szhi) szhi = p;
    }
    enum { NBLK = 512 };
    void *blk[NBLK];
    int nblk = 0;
    uintptr_t lo = (uintptr_t)-1, hi = 0;
    while (nblk < NBLK && hi < szhi + (64u << 20)) {
        void *b = malloc(64u << 20);
        memset(b, 1, pg);
        blk[nblk++] = b;
        uintptr_t p = (uintptr_t)b;
        if (p < lo) lo = p;
        if (p > hi) hi = p;
    }
    printf("szone blocks span %p .. %p\n", (void *)szlo, (void *)szhi);
    printf("tbb span (%d blks) %p .. %p\n", nblk, (void *)lo, (void *)hi);

    /* 4. R's GC over time: free most pre-proxy blocks, keep some alive. */
    int keep[NSZ], nkeep = 0, naligned = 0;
    for (int i = 0; i < NSZ; i++) {
        uintptr_t p = (uintptr_t)sz[i];
        if ((p & (pg - 1)) == 0) naligned++;
        if ((p & (pg - 1)) == 0 && p > lo && p < hi && nkeep < 16)
            keep[nkeep++] = i;
        else { free(sz[i]); sz[i] = NULL; }
    }
    malloc_zone_pressure_relief(szone, 0);
    printf("%d region-start aligned, %d also in-span kept\n", naligned, nkeep);

    /* 5. Free the survivors — any with a (now) unmapped page below is the
     * landmine. Announce before the potentially fatal free. */
    int armed = 0;
    for (int k = 0; k < nkeep; k++) {
        char *p = sz[keep[k]];
        if (page_below_unmapped(p, pg)) {
            armed++;
            printf("landmine: %p szone-owned=%s, below unmapped, in span\n",
                   (void *)p, szone->size(szone, p) > 0 ? "yes" : "no");
            printf("free()... expect EXC_BAD_ACCESS at %p\n", (void *)(p - 4));
            fflush(stdout);
        }
        free(p);
    }
    printf("SURVIVED all frees (%d armed landmines defused)\n", armed);
    if (!armed) { printf("NO LANDMINE ARMED — inconclusive run, retry\n"); return 3; }

    size_t own = malloc_size(blk[0]);
    for (int i = 0; i < nblk; i++) free(blk[i]);
    printf("tbb self-recognition: %zu; tbb frees ok\n", own);
    return 0;
}
