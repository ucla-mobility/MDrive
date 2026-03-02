#!/usr/bin/env python3
"""
Local helper server for dashboard final-review ZIP export.

Endpoint:
  POST /api/export-final-review

Input JSON body:
  {
    "records": [ ... decision records from dashboard ... ],
    "filters": {...},
    "generated_at": "...",
    "title": "..."
  }

Output:
  application/zip containing:
    review/decisions.json
    accepted/<run_name>/** route bundle files
    accepted/<run_name>/carla_validation_report.json (if available)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from carla_validation import validate_xml_manifest_contract
from setup_scenario_from_zip import parse_route_metadata


def _safe_name(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "run"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "run"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _validate_bundle_for_export(routes_dir: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    details: Dict[str, Any] = {}
    contract = validate_xml_manifest_contract(routes_dir)
    details["contract"] = contract
    if not contract.get("ok", False):
        errors.extend([str(e) for e in contract.get("errors", [])])

    manifest_path = routes_dir / "actors_manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or not manifest:
        errors.append("actors_manifest_missing_or_invalid")
        return (False, sorted(set(errors)), details)

    listed_files: List[str] = []
    existing_role_counts: Dict[str, int] = {}

    for role, entries in manifest.items():
        if not isinstance(entries, list):
            errors.append(f"manifest_role_not_list:{role}")
            continue
        existing_role_counts[str(role)] = 0
        for entry in entries:
            if not isinstance(entry, dict):
                errors.append(f"manifest_entry_not_object:{role}")
                continue
            rel = str(entry.get("file", "")).strip()
            if not rel:
                errors.append(f"manifest_entry_missing_file:{role}")
                continue
            listed_files.append(rel)
            path = (routes_dir / rel).resolve()
            if not path.exists():
                errors.append(f"manifest_file_missing:{rel}")
                continue
            if path.suffix.lower() != ".xml":
                errors.append(f"manifest_file_not_xml:{rel}")
                continue
            existing_role_counts[str(role)] += 1
            try:
                _ = parse_route_metadata(path.read_bytes())
            except Exception as exc:
                errors.append(f"parse_route_metadata_failed:{rel}:{exc}")

            if str(role) == "ego" and not re.search(r"_(\d+)\.xml$", path.name):
                errors.append(f"ego_filename_suffix_invalid:{path.name}")

    # Guardrail: role counts in manifest should match existing files.
    expected_role_counts = {str(role): len(entries) for role, entries in manifest.items() if isinstance(entries, list)}
    for role, expected_count in expected_role_counts.items():
        actual_count = int(existing_role_counts.get(role, 0))
        if actual_count != int(expected_count):
            errors.append(f"manifest_role_count_mismatch:{role}:expected={expected_count}:actual={actual_count}")

    # Guardrail: detect extra XML files not represented in manifest.
    xml_files = [
        p.relative_to(routes_dir).as_posix()
        for p in sorted(routes_dir.rglob("*.xml"))
        if p.is_file()
    ]
    listed_set = set(listed_files)
    unlisted = [rel for rel in xml_files if rel not in listed_set]
    if unlisted:
        errors.append(f"unlisted_xml_files:{len(unlisted)}")
    details["unlisted_xml_files"] = unlisted

    return (len(errors) == 0, sorted(set(errors)), details)


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    decision = str(record.get("decision", "pending") or "pending").strip().lower()
    if decision not in {"accepted", "rejected", "pending"}:
        decision = "pending"
    out = {
        "category": str(record.get("category", "") or ""),
        "run_name": str(record.get("run_name", "") or ""),
        "seed": record.get("seed", "unknown"),
        "run_dir": str(record.get("run_dir", "") or ""),
        "route_dir": str(record.get("route_dir", "") or ""),
        "route_files": record.get("route_files", []) if isinstance(record.get("route_files"), list) else [],
        "pipeline_status": str(record.get("pipeline_status", "") or ""),
        "decision": decision,
        "rejection_reason": str(record.get("rejection_reason", "") or ""),
        "reviewed_at": record.get("reviewed_at"),
        "can_accept": bool(record.get("can_accept", False)),
        "carla_validation": record.get("carla_validation", {}) if isinstance(record.get("carla_validation"), dict) else {},
    }
    return out


def _finalize_route_dir(record: Dict[str, Any]) -> Path:
    route_dir = str(record.get("route_dir", "")).strip()
    if route_dir:
        return Path(route_dir).expanduser().resolve()
    run_dir = Path(str(record.get("run_dir", "") or "")).expanduser().resolve()
    return (run_dir / "09_routes" / "routes").resolve()


def _build_zip_bytes(payload: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    raw_records = payload.get("records", [])
    if not isinstance(raw_records, list):
        raise ValueError("'records' must be a list.")

    normalized = [_normalize_record(r) for r in raw_records if isinstance(r, dict)]
    accepted = [r for r in normalized if r.get("decision") == "accepted"]
    rejected = [r for r in normalized if r.get("decision") == "rejected"]
    pending = [r for r in normalized if r.get("decision") == "pending"]

    if not accepted:
        raise ValueError("No accepted scenarios in records payload.")

    accepted_bundle_specs: List[Dict[str, Any]] = []
    validation_failures: List[Dict[str, Any]] = []
    for rec in accepted:
        route_dir = _finalize_route_dir(rec)
        run_dir = Path(str(rec.get("run_dir", "") or "")).expanduser().resolve()
        run_name = _safe_name(str(rec.get("run_name") or route_dir.parent.name or "run"))

        if not route_dir.exists() or not route_dir.is_dir():
            validation_failures.append(
                {"run_name": run_name, "error": f"route_dir_missing:{route_dir}"}
            )
            continue

        carla_validation = rec.get("carla_validation", {})
        if isinstance(carla_validation, dict):
            if not bool(carla_validation.get("passed", False)):
                reason = str(carla_validation.get("failure_reason") or "carla_gate_not_passing")
                validation_failures.append({"run_name": run_name, "error": reason})
                continue
        elif not bool(rec.get("can_accept", False)):
            validation_failures.append({"run_name": run_name, "error": "not_eligible_to_accept"})
            continue

        ok, errors, details = _validate_bundle_for_export(route_dir)
        if not ok:
            validation_failures.append({"run_name": run_name, "error": ";".join(errors), "details": details})
            continue

        report_path = run_dir / "10_carla_validation" / "output.json"
        accepted_bundle_specs.append(
            {
                "record": rec,
                "run_name": run_name,
                "run_dir": run_dir,
                "route_dir": route_dir,
                "carla_report_path": report_path if report_path.exists() else None,
            }
        )

    if validation_failures:
        raise ValueError(
            "Accepted bundle validation failed: "
            + " | ".join(
                f"{item.get('run_name', 'run')}: {item.get('error', 'unknown')}" for item in validation_failures
            )
        )

    decisions_json = {
        "generated_at": payload.get("generated_at") or datetime.now(timezone.utc).isoformat(),
        "title": payload.get("title", "Scenario Pipeline Dashboard"),
        "filters": payload.get("filters", {}),
        "counts": {
            "total_records": len(normalized),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "pending": len(pending),
        },
        "accepted_routes": accepted,
        "rejected_routes": rejected,
        "pending_routes": pending,
    }

    archive_buf = io.BytesIO()
    with zipfile.ZipFile(archive_buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "review/decisions.json",
            json.dumps(decisions_json, indent=2, ensure_ascii=False).encode("utf-8"),
        )

        for spec in accepted_bundle_specs:
            route_dir = spec["route_dir"]
            run_name = spec["run_name"]
            prefix = Path("accepted") / run_name

            for file_path in sorted(route_dir.rglob("*")):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(route_dir).as_posix()
                arcname = (prefix / rel).as_posix()
                archive.writestr(arcname, file_path.read_bytes())

            report_path = spec.get("carla_report_path")
            if report_path is not None:
                archive.writestr(
                    (prefix / "carla_validation_report.json").as_posix(),
                    Path(report_path).read_bytes(),
                )

    return archive_buf.getvalue(), decisions_json


class ExportHandler(BaseHTTPRequestHandler):
    server_version = "FinalReviewExport/1.0"

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_zip(self, data: bytes, filename: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._send_json({"ok": True})
            return
        self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/export-final-review":
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            size = int(self.headers.get("Content-Length", "0"))
        except Exception:
            size = 0
        raw = self.rfile.read(max(0, size))
        try:
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
        except Exception:
            self._send_json({"error": "Invalid JSON body."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            zip_bytes, decisions_payload = _build_zip_bytes(payload)
        except Exception as exc:
            self._send_json(
                {
                    "error": str(exc),
                    "hint": "Verify accepted runs have CARLA pass and route bundle contract compliance.",
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"final_review_bundle_{ts}.zip"
        _ = decisions_payload
        self._send_zip(zip_bytes, filename=filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8777, help="Port to bind.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, int(args.port)), ExportHandler)
    print(f"[INFO] Final review export server listening at http://{args.host}:{args.port}")
    print("[INFO] Health check: GET /api/health")
    print("[INFO] Export endpoint: POST /api/export-final-review")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
