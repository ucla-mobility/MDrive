# CARLA RPC Crash Patches

**Date**: 2026-04-13
**Target**: CARLA 0.9.12 (`CarlaUE4-Linux-Shipping`)
**Binary SHA256**: run `sha256sum carla912/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping`

## Root Cause

CARLA 0.9.12 bundles rpclib v2.2.1 (namespace `clmdep_*`) which has a
double-close race in `rpc::detail::server_session`.  The race occurs when
two threads try to close the same socket concurrently through unserialized
code paths, corrupting the ASIO epoll reactor's descriptor state and causing
a SIGSEGV in `epoll_reactor::deregister_descriptor()`.

The same bug family also exists in the streaming layer
(`carla::streaming::detail::tcp::ServerSession`).

## Patches Applied

### Binary hotpatch (LD_PRELOAD shim)

**Source**: `tools/close_ebadf_suppress.c` (tracked in git)
**Compiled**: `tools/close_ebadf_suppress.so` (built per-machine by `download_and_setup_carla.sh`)
**Mechanism**: `LD_PRELOAD` + runtime binary patching in `__attribute__((constructor))`

| Fix | What | Binary Address | Technique |
|-----|------|---------------|-----------|
| Fix 1 | `close()` EBADF suppressor | N/A (libc intercept) | Intercepts `close()` syscall wrapper; if return is `-1/EBADF`, suppresses error |
| Fix 2 | `deregister_descriptor` null-deref guard | `0x5ba0ca3` | 3-level trampoline: crash site → int3 cave → mmap'd page. Adds post-lock null check on `*descriptor_data` |
| Fix 3 | **ROOT CAUSE** – NOP out `do_read()` trailing `socket_.close()` | `0x5baa46a` | Overwrites 5-byte `call` with NOPs. Eliminates the unserialized close path that races with the close() lambda |
| Fix 4 | Guard close() lambda with `is_open()` check | `0x5baacd5` | Redirects `call basic_socket::close` through trampoline that checks `impl.socket_ != -1` before calling |

### Source patches (streaming ServerSession)

**Files modified** (not tracked by git — copies saved in this directory):
- `carla912/Plugins/carlaviz/backend/third_party/LibCarla/source/carla/streaming/detail/tcp/ServerSession.h`
- `carla912/Plugins/carlaviz/backend/third_party/LibCarla/source/carla/streaming/detail/tcp/ServerSession.cpp`

Changes:
1. Added `bool _is_closed = false;` member to `ServerSession` class
2. `CloseNow()`: added idempotency guard — returns early if `_is_closed` is true, sets it true before proceeding
3. `Close()`: added `_is_closed` check before posting `CloseNow()` to strand

## Build Instructions

`download_and_setup_carla.sh` builds the shim and patches `CarlaUE4.sh`
automatically — no manual steps required for a fresh clone. To rebuild
manually:

```bash
gcc -O2 -shared -fPIC -o tools/close_ebadf_suppress.so \
    tools/close_ebadf_suppress.c -ldl

# Verify patches activate against the binary
LD_PRELOAD=$(pwd)/tools/close_ebadf_suppress.so \
    ./carla912/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping --help 2>&1 \
    | grep '\[carla_rpc_fix\]'
# Expected: Fix 1 "active", Fix 2/3/4 each "OK"
```

## How It's Activated

`download_and_setup_carla.sh` injects an `LD_PRELOAD` block into
`carla912/CarlaUE4.sh` after each fresh tarball extraction (the block is
idempotent — re-runs are no-ops). Activation paths:
1. `carla912/CarlaUE4.sh` — unconditionally sets `LD_PRELOAD` if the `.so` exists. All MDrive tools (`hitl_start_carla.py`, `run_custom_eval.py`, `_carla_pool.py`) launch CARLA through this wrapper, so the shim activates automatically.
2. `tools/run_carla_rootcause_capture.sh` — conditionally sets `LD_PRELOAD` (default: on; `--no-ebadf-suppressor` to disable)

Note: the shim only activates if CARLA is launched via `CarlaUE4.sh`. Running the raw `CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping` binary skips the wrapper and the patches won't apply, but nothing in the repo invokes the binary that way.

## Files in This Directory

- `README.md` — this file
- `ServerSession.h.patched` — patched streaming header (inert reference; only takes effect if the carlaviz plugin is rebuilt from source)
- `ServerSession.cpp.patched` — patched streaming implementation (same)
- `ServerSession.h.original` / `ServerSession.cpp.original` — pre-patch originals for diffing
- `ServerSession.h.diff` / `ServerSession.cpp.diff` — unified diffs
- `binary_addresses.txt` — verified disassembly of all patch sites
- `verification_output.txt` — output from smoke test confirming all patches apply

The shim source itself lives at `tools/close_ebadf_suppress.c` (tracked in git).

## Upstream References

- rpclib v2.2.1 `server_session.cc`: https://github.com/rpclib/rpclib/blob/v2.2.1/lib/rpc/detail/server_session.cc
- rpclib `async_writer.h`: https://github.com/rpclib/rpclib/blob/v2.2.1/include/rpc/detail/async_writer.h
- CARLA GitHub issue (same bug family): https://github.com/carla-simulator/carla/issues/6209
