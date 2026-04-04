#!/usr/bin/env python

"""
Checkpoint persistence for evaluator-side crash recovery.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckpointRef:
    index: int
    path: Path
    meta: dict[str, Any]


@dataclass
class CheckpointCommitResult:
    ok: bool
    checkpoint_id: int | None
    ref: CheckpointRef | None
    error_type: str = ""
    error: str = ""
    snapshot_prepare_ms: float = 0.0
    checkpoint_write_ms: float = 0.0
    checkpoint_payload_bytes: int = 0
    checkpoint_queue_depth: int = 0
    skipped_checkpoints_due_to_backpressure: int = 0


class CheckpointManager:
    def __init__(
        self,
        run_root: str | Path,
        *,
        checkpoint_interval_frames: int = 20,
        max_checkpoints: int = 0,
        async_writes: bool = True,
        queue_size: int = 1,
    ) -> None:
        self.run_root = Path(run_root)
        self.checkpoints_dir = self.run_root / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.interval_frames = max(1, int(checkpoint_interval_frames))
        self.max_checkpoints = max(0, int(max_checkpoints))
        self._last_checkpoint_frame = -1
        self.async_writes = bool(async_writes)
        self._lock = threading.Lock()
        self._backpressure_skips = 0
        self._stop_event = threading.Event()
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._completed: list[CheckpointCommitResult] = []

        self._cleanup_incomplete_checkpoints()
        self._next_index = self._discover_next_index()
        self._worker_thread: threading.Thread | None = None
        if self.async_writes:
            self._worker_thread = threading.Thread(
                target=self._writer_loop,
                name="carla_checkpoint_writer",
                daemon=True,
            )
            self._worker_thread.start()

    @staticmethod
    def _fsync_path(path: Path) -> None:
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except Exception:
            return
        try:
            os.fsync(fd)
        except Exception:
            pass
        finally:
            try:
                os.close(fd)
            except Exception:
                pass

    def _cleanup_incomplete_checkpoints(self) -> None:
        # Remove stale write-in-progress directories from previous crashes.
        for incomplete_dir in sorted(self.checkpoints_dir.glob(".ckpt_*.incomplete")):
            try:
                for child in sorted(incomplete_dir.glob("*")):
                    child.unlink(missing_ok=True)
                incomplete_dir.rmdir()
            except Exception:
                continue

        # Ignore and clean malformed checkpoint directories left by interrupted writes.
        for ckpt_dir in sorted(self.checkpoints_dir.glob("ckpt_*")):
            if not ckpt_dir.is_dir():
                continue
            payload_path = ckpt_dir / "payload.pkl.gz"
            meta_path = ckpt_dir / "meta.json"
            if payload_path.exists() and meta_path.exists():
                # Best-effort cleanup of stale temp files inside otherwise valid checkpoints.
                for tmp in ckpt_dir.glob("*.tmp"):
                    tmp.unlink(missing_ok=True)
                continue
            try:
                for child in sorted(ckpt_dir.glob("*")):
                    child.unlink(missing_ok=True)
                ckpt_dir.rmdir()
            except Exception:
                continue

    def _discover_next_index(self) -> int:
        max_index = 0
        for path in self.checkpoints_dir.glob("ckpt_*"):
            try:
                idx = int(path.name.split("_", 1)[1])
            except Exception:
                continue
            max_index = max(max_index, idx)
        return max_index + 1

    def should_checkpoint(self, logical_frame_id: int) -> bool:
        logical_frame_id = int(logical_frame_id)
        if logical_frame_id < 0:
            return False
        if self._last_checkpoint_frame < 0:
            return True
        return (logical_frame_id - self._last_checkpoint_frame) >= self.interval_frames

    def _reserve_checkpoint(self, *, logical_frame_id: int, route_name: str, route_index: int, crash_generation: int, segment_id: int, reason: str) -> tuple[int, dict[str, Any]]:
        with self._lock:
            index = int(self._next_index)
            self._next_index = index + 1
        meta = {
            "checkpoint_id": index,
            "created_unix_time": time.time(),
            "logical_frame_id": int(logical_frame_id),
            "route_name": str(route_name or ""),
            "route_index": int(route_index),
            "crash_generation": int(crash_generation),
            "segment_id": int(segment_id),
            "reason": str(reason or "periodic"),
        }
        return index, meta

    def _write_checkpoint(self, *, index: int, snapshot: dict[str, Any], meta: dict[str, Any]) -> tuple[CheckpointRef, int]:
        ckpt_dir = self.checkpoints_dir / f"ckpt_{int(index):06d}"
        inprogress_dir = self.checkpoints_dir / f".ckpt_{int(index):06d}.incomplete"
        if inprogress_dir.exists():
            for child in sorted(inprogress_dir.glob("*")):
                child.unlink(missing_ok=True)
            inprogress_dir.rmdir()
        inprogress_dir.mkdir(parents=True, exist_ok=False)
        payload_tmp = inprogress_dir / "payload.pkl.gz.tmp"
        payload_path = inprogress_dir / "payload.pkl.gz"
        meta_tmp = inprogress_dir / "meta.json.tmp"
        meta_path = inprogress_dir / "meta.json"

        with payload_tmp.open("wb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb") as gzip_handle:
                pickle.dump(snapshot, gzip_handle, protocol=pickle.HIGHEST_PROTOCOL)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        payload_tmp.replace(payload_path)

        with meta_tmp.open("w", encoding="utf-8") as meta_handle:
            json.dump(meta, meta_handle, indent=2, sort_keys=True)
            meta_handle.flush()
            os.fsync(meta_handle.fileno())
        meta_tmp.replace(meta_path)

        self._fsync_path(inprogress_dir)
        inprogress_dir.replace(ckpt_dir)
        self._fsync_path(self.checkpoints_dir)

        payload_bytes = int((ckpt_dir / "payload.pkl.gz").stat().st_size)
        ref = CheckpointRef(index=int(index), path=ckpt_dir, meta=meta)
        self._enforce_retention()
        return ref, payload_bytes

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if request is None:
                self._queue.task_done()
                break

            checkpoint_id = int(request.get("checkpoint_id", -1))
            snapshot_prepare_ms = float(request.get("snapshot_prepare_ms", 0.0) or 0.0)
            queue_depth = int(self._queue.qsize())
            start = time.perf_counter()
            payload_bytes = 0
            ref: CheckpointRef | None = None
            error_type = ""
            error = ""
            ok = False
            try:
                ref, payload_bytes = self._write_checkpoint(
                    index=checkpoint_id,
                    snapshot=request.get("snapshot") or {},
                    meta=request.get("meta") or {},
                )
                ok = True
            except Exception as exc:
                error_type = type(exc).__name__
                error = str(exc)
            write_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
            with self._lock:
                skipped = int(self._backpressure_skips)
                self._completed.append(
                    CheckpointCommitResult(
                        ok=ok,
                        checkpoint_id=checkpoint_id if checkpoint_id >= 0 else None,
                        ref=ref,
                        error_type=error_type,
                        error=error,
                        snapshot_prepare_ms=snapshot_prepare_ms,
                        checkpoint_write_ms=write_ms,
                        checkpoint_payload_bytes=int(payload_bytes),
                        checkpoint_queue_depth=queue_depth,
                        skipped_checkpoints_due_to_backpressure=skipped,
                    )
                )
            self._queue.task_done()

    def enqueue_commit(
        self,
        snapshot: dict[str, Any],
        *,
        logical_frame_id: int,
        route_name: str,
        route_index: int,
        crash_generation: int,
        segment_id: int,
        reason: str = "periodic",
        snapshot_prepare_ms: float = 0.0,
    ) -> dict[str, Any]:
        logical_frame_id = int(logical_frame_id)
        queue_depth = int(self._queue.qsize()) if self.async_writes else 0
        if self.async_writes and queue_depth >= self._queue.maxsize:
            with self._lock:
                self._backpressure_skips += 1
                skipped = int(self._backpressure_skips)
            self._last_checkpoint_frame = logical_frame_id
            return {
                "scheduled": False,
                "checkpoint_id": None,
                "queue_depth": queue_depth,
                "skipped_checkpoints_due_to_backpressure": skipped,
            }

        checkpoint_id, meta = self._reserve_checkpoint(
            logical_frame_id=logical_frame_id,
            route_name=route_name,
            route_index=route_index,
            crash_generation=crash_generation,
            segment_id=segment_id,
            reason=reason,
        )
        self._last_checkpoint_frame = logical_frame_id

        if not self.async_writes:
            start = time.perf_counter()
            try:
                ref, payload_bytes = self._write_checkpoint(index=checkpoint_id, snapshot=snapshot, meta=meta)
                write_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
                with self._lock:
                    self._completed.append(
                        CheckpointCommitResult(
                            ok=True,
                            checkpoint_id=checkpoint_id,
                            ref=ref,
                            snapshot_prepare_ms=float(snapshot_prepare_ms),
                            checkpoint_write_ms=write_ms,
                            checkpoint_payload_bytes=int(payload_bytes),
                            checkpoint_queue_depth=0,
                            skipped_checkpoints_due_to_backpressure=int(self._backpressure_skips),
                        )
                    )
            except Exception as exc:
                with self._lock:
                    self._completed.append(
                        CheckpointCommitResult(
                            ok=False,
                            checkpoint_id=checkpoint_id,
                            ref=None,
                            error_type=type(exc).__name__,
                            error=str(exc),
                            snapshot_prepare_ms=float(snapshot_prepare_ms),
                            checkpoint_queue_depth=0,
                            skipped_checkpoints_due_to_backpressure=int(self._backpressure_skips),
                        )
                    )
            return {
                "scheduled": True,
                "checkpoint_id": checkpoint_id,
                "queue_depth": 0,
                "skipped_checkpoints_due_to_backpressure": int(self._backpressure_skips),
            }

        self._queue.put_nowait(
            {
                "checkpoint_id": checkpoint_id,
                "snapshot": snapshot,
                "meta": meta,
                "snapshot_prepare_ms": float(snapshot_prepare_ms or 0.0),
            }
        )
        return {
            "scheduled": True,
            "checkpoint_id": checkpoint_id,
            "queue_depth": int(self._queue.qsize()),
            "skipped_checkpoints_due_to_backpressure": int(self._backpressure_skips),
        }

    def drain_completed(self) -> list[CheckpointCommitResult]:
        with self._lock:
            results = list(self._completed)
            self._completed.clear()
        return results

    def commit(
        self,
        snapshot: dict[str, Any],
        *,
        logical_frame_id: int,
        route_name: str,
        route_index: int,
        crash_generation: int,
        segment_id: int,
        reason: str = "periodic",
    ) -> CheckpointRef:
        index, meta = self._reserve_checkpoint(
            logical_frame_id=int(logical_frame_id),
            route_name=route_name,
            route_index=int(route_index),
            crash_generation=int(crash_generation),
            segment_id=int(segment_id),
            reason=reason,
        )
        ref, _payload_bytes = self._write_checkpoint(index=index, snapshot=snapshot, meta=meta)
        self._last_checkpoint_frame = int(logical_frame_id)
        return ref

    def _enforce_retention(self) -> None:
        if self.max_checkpoints <= 0:
            return
        refs = self.list_checkpoints()
        if len(refs) <= self.max_checkpoints:
            return
        overflow = len(refs) - self.max_checkpoints
        for ref in refs[:overflow]:
            try:
                for child in sorted(ref.path.glob("*")):
                    child.unlink(missing_ok=True)
                ref.path.rmdir()
            except Exception:
                continue

    def list_checkpoints(self) -> list[CheckpointRef]:
        refs: list[CheckpointRef] = []
        for ckpt_dir in sorted(self.checkpoints_dir.glob("ckpt_*")):
            meta_path = ckpt_dir / "meta.json"
            payload_path = ckpt_dir / "payload.pkl.gz"
            if not meta_path.exists() or not payload_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                checkpoint_id = int(meta.get("checkpoint_id"))
            except Exception:
                continue
            refs.append(CheckpointRef(index=checkpoint_id, path=ckpt_dir, meta=meta))
        refs.sort(key=lambda ref: ref.index)
        return refs

    def latest_checkpoint(self) -> CheckpointRef | None:
        refs = self.list_checkpoints()
        if not refs:
            return None
        return refs[-1]

    def load_payload(self, ref: CheckpointRef) -> dict[str, Any] | None:
        payload_path = ref.path / "payload.pkl.gz"
        if not payload_path.exists():
            return None
        try:
            with gzip.open(payload_path, "rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict):
                return payload
            return None
        except Exception:
            return None

    def load_latest_payload(self) -> tuple[CheckpointRef, dict[str, Any]] | tuple[None, None]:
        ref = self.latest_checkpoint()
        if ref is None:
            return None, None
        payload = self.load_payload(ref)
        if payload is None:
            return None, None
        return ref, payload

    def close(self) -> None:
        if not self.async_writes:
            return
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
