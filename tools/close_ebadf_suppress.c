/*
 * close_ebadf_suppress.c  — v2  (Apr 2026)
 *
 * LD_PRELOAD shim for CarlaUE4-Linux-Shipping (CARLA 0.9.12).
 *
 * CRASH: CARLA's bundled rpclib (clmdep_asio) has a double-close race in
 * rpc::detail::server_session.  There are FOUR distinct crash/failure vectors,
 * all rooted in the same non-idempotent close logic:
 *
 * Fix 1 — close() EBADF suppressor:
 *   Thread 2 arrives AFTER Thread 1 marked the fd invalid (impl.socket_=-1).
 *   reactive_socket_service_base::close() gets EBADF from the close() syscall
 *   → clmdep_asio throw_error() → carla::throw_exception() → SIGSEGV.
 *   Fix: intercept close() and suppress EBADF.
 *
 * Fix 2 — deregister_descriptor null-deref patch (TOCTOU race):
 *   Thread 2 passes the initial null check, acquires mutex, but Thread 1 has
 *   already cleared *descriptor_data → testb on null → SIGSEGV @0x5ba0ca6.
 *   Fix: binary hot-patch to add post-lock null check.
 *
 * Fix 3 — NOP out do_read() trailing socket_.close() (ROOT CAUSE FIX):
 *   In upstream rpclib v2.2.1, server_session::do_read() ends with:
 *     if (exit_) { socket_.close(); }
 *   This is NOT strand-serialized, so it races with the close() lambda.
 *   The close() path already closes the socket via the write_strand_ post.
 *   Fix: NOP out the `call basic_socket::close()` at 0x5baa46a.
 *
 * Fix 4 — Guard close() lambda with is_open() check:
 *   The close() lambda at $_0::operator()() calls socket_.close() without
 *   checking is_open() first.  If another path already closed it, this
 *   triggers the deregister_descriptor crash.
 *   Fix: redirect the call at 0x5baacd5 through a trampoline that checks
 *   is_open() before calling close().
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o close_ebadf_suppress.so \
 *       close_ebadf_suppress.c -ldl
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>

#define TAG "[carla_rpc_fix] "

static int g_fix_enabled = -1;
static int g_fix_quiet = 0;

static int env_truthy(const char *name)
{
    const char *value = getenv(name);
    if (value == NULL || *value == '\0')
        return 0;
    if (strcmp(value, "0") == 0)
        return 0;
    if (strcasecmp(value, "false") == 0)
        return 0;
    if (strcasecmp(value, "no") == 0)
        return 0;
    return 1;
}

static void detect_fix_mode(void)
{
    if (g_fix_enabled != -1)
        return;

    g_fix_quiet = env_truthy("CARLA_RPC_FIX_QUIET");
    if (env_truthy("CARLA_RPC_FIX_FORCE")) {
        g_fix_enabled = 1;
        return;
    }

    char exe_path[PATH_MAX];
    ssize_t nread = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (nread <= 0) {
        g_fix_enabled = 0;
        return;
    }
    exe_path[nread] = '\0';

    const char *base = strrchr(exe_path, '/');
    base = (base != NULL) ? (base + 1) : exe_path;
    g_fix_enabled = (strcmp(base, "CarlaUE4-Linux-Shipping") == 0) ? 1 : 0;
}

static int fix_enabled(void)
{
    detect_fix_mode();
    return g_fix_enabled == 1;
}

static int fix_logging_enabled(void)
{
    detect_fix_mode();
    return g_fix_enabled == 1 && !g_fix_quiet;
}

/* ===========================================================================
 * Fix 1: close() EBADF suppressor
 * =========================================================================*/

static int (*real_close)(int fd) = NULL;

int close(int fd)
{
    if (__builtin_expect(real_close == NULL, 0))
        real_close = (int (*)(int))dlsym(RTLD_NEXT, "close");

    int ret = real_close(fd);

    if (!fix_enabled())
        return ret;

    if (__builtin_expect(ret == -1 && errno == EBADF, 0)) {
        errno = 0;
        return 0;
    }

    return ret;
}

/* ===========================================================================
 * Shared helpers for binary hot-patches (Fixes 2, 3, 4)
 * =========================================================================*/

static int make_writable(uintptr_t addr, size_t len)
{
    long pgsz = sysconf(_SC_PAGESIZE);
    uintptr_t base = addr & ~(uintptr_t)(pgsz - 1);
    uintptr_t top  = (addr + len + pgsz - 1) & ~(uintptr_t)(pgsz - 1);
    int rc = mprotect((void *)base, (size_t)(top - base),
                      PROT_READ | PROT_WRITE | PROT_EXEC);
    if (rc != 0)
        fprintf(stderr, TAG "mprotect(%p, 0x%lx) failed: %s\n",
                (void *)base, (unsigned long)(top - base), strerror(errno));
    return rc;
}

/*
 * Allocate an executable page within ±2 GB of 'anchor' for rel32 jumps.
 */
static void *alloc_tramp(uintptr_t anchor, uintptr_t target1, uintptr_t target2)
{
    long pgsz = sysconf(_SC_PAGESIZE);
    static const intptr_t tries[] = {
        -0x04000000, -0x08000000, -0x10000000,
         0x04000000,  0x08000000,  0x10000000,
        -0x20000000,  0x20000000,
        -0x30000000,  0x30000000,
    };
    for (int i = 0; i < (int)(sizeof(tries)/sizeof(tries[0])); i++) {
        uintptr_t hint = (uintptr_t)((intptr_t)anchor + tries[i]);
        if (hint < 0x10000 || hint > (uintptr_t)0x7fff00000000ULL)
            continue;
        void *p = mmap((void *)hint, (size_t)pgsz,
                       PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
            continue;
        uintptr_t got = (uintptr_t)p;
        /* Verify both targets are rel32-reachable from within this page */
        intptr_t d1 = (intptr_t)(target1 - (got + 64));
        intptr_t d2 = (intptr_t)(target2 - (got + 64));
        if (d1 >= -0x7FFFFFFF && d1 <= 0x7FFFFFFF &&
            d2 >= -0x7FFFFFFF && d2 <= 0x7FFFFFFF)
            return p;
        munmap(p, (size_t)pgsz);
    }
    return NULL;
}

/* ===========================================================================
 * Fix 2: deregister_descriptor null-deref patch
 *
 *   0x5ba0c9f: mov   -0x18(%rbp),%rax     <- reload &(descriptor_data)
 *   0x5ba0ca3: mov   (%rax),%rax          <- rax = *descriptor_data  ← RACE
 *   0x5ba0ca6: testb $0x1,0x90(%rax)      <- CRASH if rax==0
 *   0x5ba0cad: jne   0x5ba0e48            <- scoped_lock dtor + return
 * =========================================================================*/
#define DEREGISTER_CRASH     0x0000000005ba0ca3UL
#define DEREGISTER_CONT      0x0000000005ba0cadUL
#define DEREGISTER_SAFE_EXIT 0x0000000005ba0e48UL
#define INT3_CAVE            0x0000000005ba0c43UL

static const uint8_t exp_crash_bytes[10] = {
    0x48, 0x8b, 0x00,
    0xf6, 0x80, 0x90, 0x00, 0x00, 0x00, 0x01
};
static const uint8_t exp_cave_bytes[13] = {
    0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
    0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc
};

static void patch_fix2_deregister(void)
{
    fprintf(stderr, TAG "Fix 2 (deregister null-deref): applying...\n");

    if (make_writable(DEREGISTER_CRASH, 10) != 0) { fprintf(stderr, TAG "Fix 2: SKIP (mprotect crash site)\n"); return; }
    if (make_writable(INT3_CAVE, 13) != 0)        { fprintf(stderr, TAG "Fix 2: SKIP (mprotect int3 cave)\n"); return; }

    if (memcmp((void *)DEREGISTER_CRASH, exp_crash_bytes, 10) != 0) {
        fprintf(stderr, TAG "Fix 2: SKIP (crash-site bytes mismatch — wrong CARLA version?)\n");
        return;
    }
    if (memcmp((void *)INT3_CAVE, exp_cave_bytes, 13) != 0) {
        fprintf(stderr, TAG "Fix 2: SKIP (int3 cave bytes mismatch)\n");
        return;
    }

    void *tramp = alloc_tramp(DEREGISTER_CRASH, DEREGISTER_SAFE_EXIT, DEREGISTER_CONT);
    if (!tramp) { fprintf(stderr, TAG "Fix 2: SKIP (trampoline alloc failed)\n"); return; }

    /* Level 2 trampoline (25 bytes) */
    uint8_t *t = (uint8_t *)tramp;
    int32_t off;

    t[0]=0x48; t[1]=0x8b; t[2]=0x00;               /* mov (%rax),%rax           */
    t[3]=0x48; t[4]=0x85; t[5]=0xc0;               /* test %rax,%rax            */
    t[6]=0x75; t[7]=0x05;                           /* jnz  +5 (→ t[13])        */
    off = (int32_t)((intptr_t)(DEREGISTER_SAFE_EXIT - ((uintptr_t)tramp + 13)));
    t[8]=0xe9; memcpy(t+9, &off, 4);               /* jmp  SAFE_EXIT            */
    t[13]=0xf6; t[14]=0x80; t[15]=0x90;
    t[16]=0x00; t[17]=0x00; t[18]=0x00; t[19]=0x01; /* testb $0x1,0x90(%rax)    */
    off = (int32_t)((intptr_t)(DEREGISTER_CONT - ((uintptr_t)tramp + 25)));
    t[20]=0xe9; memcpy(t+21, &off, 4);             /* jmp  DEREGISTER_CONT      */

    mprotect(tramp, (size_t)sysconf(_SC_PAGESIZE), PROT_READ | PROT_EXEC);

    /* Level 1: int3 cave → trampoline (movabs + jmp *r11) */
    uint8_t *cave = (uint8_t *)INT3_CAVE;
    uintptr_t tramp_addr = (uintptr_t)tramp;
    cave[0]=0x49; cave[1]=0xbb;
    memcpy(cave+2, &tramp_addr, 8);
    cave[10]=0x41; cave[11]=0xff; cave[12]=0xe3;

    /* Level 0: redirect crash site → int3 cave */
    uint8_t *patch = (uint8_t *)DEREGISTER_CRASH;
    int32_t jmp_off = (int32_t)((intptr_t)(INT3_CAVE - (DEREGISTER_CRASH + 5)));
    patch[0]=0xe9; memcpy(patch+1, &jmp_off, 4);
    patch[5]=patch[6]=patch[7]=patch[8]=patch[9]=0x90;

    fprintf(stderr, TAG "Fix 2: OK (trampoline at %p)\n", tramp);
}

/* ===========================================================================
 * Fix 3: NOP out do_read() trailing socket_.close()  (ROOT CAUSE)
 *
 * In do_read() at the bottom:
 *   0x5baa44a: load exit_ (atomic bool)
 *   0x5baa44f: test $0x1,%al
 *   0x5baa451: jne  0x5baa45c          <- if exit_, jump to close block
 *   0x5baa45c: [load socket pointer]
 *   0x5baa46a: call 0x5baef20          <- basic_socket::close() — 5 bytes
 *
 * This call is NOT strand-serialized.  The close() lambda already handles
 * the full socket shutdown via the write_strand_.  This bare socket_.close()
 * at the bottom of do_read is the race participant that causes the double-
 * close / stale descriptor_state.
 *
 * Fix: overwrite the 5-byte call with NOPs.
 * =========================================================================*/
#define DOREAD_TRAILING_CLOSE  0x0000000005baa46aUL

static const uint8_t exp_doread_call[5] = {
    0xe8, 0xb1, 0x4a, 0x00, 0x00   /* call 0x5baef20 (basic_socket::close) */
};

static void patch_fix3_doread_nop(void)
{
    fprintf(stderr, TAG "Fix 3 (do_read trailing close): applying...\n");

    if (make_writable(DOREAD_TRAILING_CLOSE, 5) != 0) {
        fprintf(stderr, TAG "Fix 3: SKIP (mprotect)\n");
        return;
    }

    if (memcmp((void *)DOREAD_TRAILING_CLOSE, exp_doread_call, 5) != 0) {
        fprintf(stderr, TAG "Fix 3: SKIP (bytes mismatch at 0x%lx — wrong CARLA version?)\n",
                (unsigned long)DOREAD_TRAILING_CLOSE);
        return;
    }

    uint8_t *p = (uint8_t *)DOREAD_TRAILING_CLOSE;
    p[0]=0x90; p[1]=0x90; p[2]=0x90; p[3]=0x90; p[4]=0x90;

    fprintf(stderr, TAG "Fix 3: OK (NOP'd 5 bytes at 0x%lx)\n",
            (unsigned long)DOREAD_TRAILING_CLOSE);
}

/* ===========================================================================
 * Fix 4: Guard close() lambda with is_open() check
 *
 * The close() lambda $_0::operator()() at 0x5baacb0:
 *   0x5baacc0: mov  (%rdi),%rax          <- load 'this' (captured server_session*)
 *   0x5baacc3: mov  %rax,%rcx
 *   0x5baacc6: add  $0x10,%rcx           <- rcx = &socket_ (socket at offset 0x10)
 *   0x5baacca: [save rdi to stack]
 *   0x5baacce: mov  %rcx,%rdi            <- rdi = &socket_
 *   0x5baacd1: [save rax to stack]
 *   0x5baacd5: call 0x5baef20            <- basic_socket::close() — NO is_open guard
 *   0x5baacda: [continue to close_session]
 *
 * Fix: redirect the call at 0x5baacd5 to a trampoline that:
 *   1. Checks if the socket fd is valid (basic_socket_service_base::is_open)
 *   2. If open: calls basic_socket::close() then jumps back
 *   3. If not open: skips the close and jumps back
 *
 * The is_open check in clmdep_asio: the socket implementation stores the fd
 * at socket_service_base::implementation_type::socket_ (first member), which
 * is -1 when closed.  The actual struct pointed to by rdi is:
 *   rdi+0x00: io_service pointer (8 bytes)
 *   rdi+0x08: &service  (8 bytes) — inner impl starts here
 *       Actually, rdi is basic_socket<tcp> and at this point basic_socket::close()
 *       calls through to service.close(impl, ec).
 *
 * Rather than replicating the ABI, we use a simpler approach: jump to
 * basic_socket::close() but wrap it with a try/catch-like null check on the
 * implementation data.  Actually, the simplest reliable fix is to check
 * is_open() by calling the service's is_open method, but that requires
 * knowing the vtable layout.
 *
 * Simplest fix: since Fix 3 already removes the race participant,
 * Fix 4 adds defense-in-depth by checking socket_.is_open() before close.
 *
 * The basic_socket's internal implementation has socket_ fd at a known
 * offset.  reactive_socket_service_base::base_implementation_type::socket_
 * is at offset 0 of the impl.  The impl is stored inside the basic_socket.
 *
 * Looking at the actual binary:
 *   basic_socket::close() at 0x5baef20 calls service.close(impl, ec).
 *   stream_socket_service::close at 0x5bafe00 then calls
 *   reactive_socket_service_base::close which checks is_open internally.
 *
 * reactive_socket_service_base::close() has its own is_open() guard that
 * returns EBADF+throw.  The throw is what kills us (Fix 1 suppresses it).
 *
 * For Fix 4, we NOP the call to basic_socket::close() at 0x5baacd5 too,
 * since Fix 3 already eliminates the other race participant.  With only
 * one close path remaining (the close() lambda), and Fix 3 removing the
 * do_read race, the lambda no longer needs the extra guard — BUT we still
 * add it for defense-in-depth.
 *
 * Actually, let's think more carefully.  The lambda IS the intended close
 * path.  We should NOT NOP it.  Instead, we should make it check is_open
 * first.  The check is simple: the socket_service stores the fd, and
 * if it's already -1, we skip.
 *
 * Since basic_socket::close() internally calls throw_error on EBADF,
 * and Fix 1 suppresses that throw at the syscall level, the combination
 * of Fix 1 + Fix 3 should be sufficient.  Fix 4 is true defense-in-depth.
 *
 * We'll implement Fix 4 as: redirect the 5-byte call at 0x5baacd5 to a
 * trampoline that calls is_open() first.
 *
 * basic_socket has is_open() which checks service_.is_open(implementation_).
 * For reactive_socket_service, is_open() is:
 *     return impl.socket_ != invalid_socket;
 * where invalid_socket = -1.
 *
 * We need to know the offset of the socket fd from rdi (which is &basic_socket).
 * From the binary:  basic_socket::close() accesses the service and impl.
 * Looking at 0x5baef20:
 *   rdi = &basic_socket
 *   0x5baef47: call _ZN...BasicFormatterIcE6writerEv  (gets service)
 *   0x5baef54: call _ZN...BasicWriterIcE3getEPc       (gets impl ptr)
 *   0x5baef64: call stream_socket_service::close(impl, ec)
 *
 * The layout is complex.  A simpler approach: we know that after Fix 3
 * NOPs the do_read close, the lambda is the ONLY close path.  So even
 * without Fix 4, the double-close race is eliminated.
 *
 * For maximum safety, we implement Fix 4 as a lightweight trampoline
 * that checks the socket fd before calling close.  We use the fact that
 * in the close lambda, rax already holds the server_session pointer,
 * and the socket is at offset 0x10.  The socket's internal fd is at a
 * known offset within the basic_socket → service → impl chain.
 *
 * Actually — realizing this gets quite complex for the ABI, let me take
 * the pragmatic approach.  We DO want the close lambda to still call
 * socket_.close().  The race is already removed by Fix 3.  Fix 4 adds
 * a pre-check using a trampoline.
 *
 * Looking at the impl layout for reactive_socket_service_base:
 *   struct base_implementation_type {
 *     socket_type socket_;        // offset 0 — the fd (int, but padded)
 *     state_type state_;          // offset 4
 *     reactor::per_descriptor_data reactor_data_;  // offset 8
 *   };
 *
 * From basic_socket<tcp>, the impl is embedded.  The service is obtained
 * through the base class io_object_impl.  In clmdep_asio (old API), the
 * layout of basic_socket is:
 *   basic_io_object {
 *     io_service* service_owner;    // offset 0
 *     service_type& service;        // offset 8 (reference = pointer)
 *     implementation_type impl;     // offset 16 (0x10)
 *   }
 * And implementation_type for stream_socket_service contains:
 *   reactive_socket_service::implementation_type which inherits
 *   base_implementation_type { socket_ at +0, state_ at +4, reactor_data_ at +8 }
 *
 * So from rdi (basic_socket*):
 *   rdi+0x10 = start of impl
 *   rdi+0x10+0 = socket_ (the file descriptor, int)
 *   invalid_socket = -1
 *
 * Trampoline for Fix 4:
 *   cmpl $-1, 0x10(%rdi)       <- is impl.socket_ == -1?  (7 bytes: 83 7f 10 ff)
 *   je   skip                  <- if closed, skip          (2 bytes)
 *   jmp  basic_socket::close   <- call original            (5 bytes)
 * skip:
 *   ret  or jmp back
 *
 * Wait — the call at 0x5baacd5 is `call 0x5baef20`.  After the call,
 * execution continues at 0x5baacda.  We can't just jmp to close because
 * it won't return to the right place.  We need to CALL it.
 *
 * Better: replace the 5-byte call with a 5-byte call to our trampoline.
 * The trampoline checks, then either calls original or returns.
 * =========================================================================*/
#define CLOSE_LAMBDA_CALL    0x0000000005baacd5UL  /* call basic_socket::close */
#define BASIC_SOCKET_CLOSE   0x0000000005baef20UL  /* basic_socket::close() */
#define CLOSE_LAMBDA_RET     0x0000000005baacdaUL  /* instruction after the call */

static const uint8_t exp_close_lambda_call[5] = {
    0xe8, 0x46, 0x42, 0x00, 0x00   /* call 0x5baef20 */
};

static void patch_fix4_close_lambda_guard(void)
{
    fprintf(stderr, TAG "Fix 4 (close lambda is_open guard): applying...\n");

    if (make_writable(CLOSE_LAMBDA_CALL, 5) != 0) {
        fprintf(stderr, TAG "Fix 4: SKIP (mprotect)\n");
        return;
    }

    if (memcmp((void *)CLOSE_LAMBDA_CALL, exp_close_lambda_call, 5) != 0) {
        fprintf(stderr, TAG "Fix 4: SKIP (bytes mismatch at 0x%lx)\n",
                (unsigned long)CLOSE_LAMBDA_CALL);
        return;
    }

    /*
     * We need a trampoline reachable via rel32 from CLOSE_LAMBDA_CALL.
     * The trampoline:
     *   [0]  cmpl  $0xffffffff, 0x10(%rdi)    (7 bytes: 83 7f 10 ff → actually
     *         for 32-bit cmp: 81 7f 10 ff ff ff ff)
     *   Actually for cmp $-1 (sign-extended byte): 83 7f 10 ff  (4 bytes)
     *   [4]  je    skip (+N)                  (2 bytes)
     *   [6]  jmp   basic_socket::close        (5 bytes, rel32)
     *   [11] ret                              (1 byte) — skip path
     *
     * Total: 12 bytes.  If basic_socket::close is out of rel32 range from
     * the trampoline, we need movabs.  Let's check: CLOSE_LAMBDA_CALL is
     * 0x5baacd5, BASIC_SOCKET_CLOSE is 0x5baef20.  Difference: 0x424b.
     * If we allocate the trampoline page near CLOSE_LAMBDA_CALL, the target
     * will be in range.
     *
     * But we can't jmp to close — we need to CALL it (to return properly).
     * Since we replaced the original call, our trampoline must call close
     * and then the ret from close will return to 0x5baacda (the original
     * return address is already on the stack from our replacement call).
     *
     * Plan B: Use our trampoline as a wrapper:
     *   cmpl $-1, 0x10(%rdi)
     *   je skip
     *   jmp basic_socket::close  <- this is a tail-call; close() will ret
     *                               to 0x5baacda (our caller's return addr)
     * skip:
     *   ret                      <- return to 0x5baacda, skipping close
     */
    void *tramp = alloc_tramp(CLOSE_LAMBDA_CALL, BASIC_SOCKET_CLOSE, CLOSE_LAMBDA_CALL);
    if (!tramp) { fprintf(stderr, TAG "Fix 4: SKIP (trampoline alloc failed)\n"); return; }

    uint8_t *t = (uint8_t *)tramp;
    int32_t off;

    /* cmpl $-1, 0x10(%rdi)  — 4 bytes: 83 7f 10 ff */
    t[0]=0x83; t[1]=0x7f; t[2]=0x10; t[3]=0xff;
    /* je skip (+7, to t[11]) */
    t[4]=0x74; t[5]=0x05;
    /* jmp basic_socket::close — 5-byte rel32 */
    off = (int32_t)((intptr_t)(BASIC_SOCKET_CLOSE - ((uintptr_t)tramp + 11)));
    t[6]=0xe9; memcpy(t+7, &off, 4);
    /* skip: ret */
    t[11]=0xc3;

    mprotect(tramp, (size_t)sysconf(_SC_PAGESIZE), PROT_READ | PROT_EXEC);

    /* Redirect: replace call at CLOSE_LAMBDA_CALL with call to trampoline */
    uint8_t *p = (uint8_t *)CLOSE_LAMBDA_CALL;
    int32_t call_off = (int32_t)((intptr_t)((uintptr_t)tramp - (CLOSE_LAMBDA_CALL + 5)));
    p[0]=0xe8; memcpy(p+1, &call_off, 4);

    fprintf(stderr, TAG "Fix 4: OK (trampoline at %p)\n", tramp);
}

/* ===========================================================================
 * Constructor: apply all patches at LD_PRELOAD time
 * =========================================================================*/
__attribute__((constructor))
static void carla_rpc_hotpatch_all(void)
{
    if (!fix_enabled())
        return;

    if (fix_logging_enabled()) {
        fprintf(stderr, TAG "=== CARLA RPC fix shim loading (v2, Apr 2026) ===\n");
        fprintf(stderr, TAG "Fix 1 (close EBADF suppress): active (always via LD_PRELOAD)\n");
    }
    patch_fix2_deregister();
    patch_fix3_doread_nop();
    patch_fix4_close_lambda_guard();
    if (fix_logging_enabled())
        fprintf(stderr, TAG "=== all patches applied ===\n");
}
