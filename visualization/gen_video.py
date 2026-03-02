import argparse
import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
STEP_TIME_PATTERN = re.compile(r"step infer time:\s*([0-9.]+)")
FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL_PADDING = 14
VEHICLE_DIR_PATTERN = re.compile(r"^(?:ego_vehicle_|rgb_|meta_)(\d+)$")
ALLOWED_STREAM_PREFIXES = ("ego_vehicle_", "rgb_", "meta_", "1", "2")


def _parse_suffix_filter(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    parts = [p.strip().lstrip("_") for p in re.split(r"[,\s]+", raw) if p.strip()]
    return [p.lower() for p in parts if p]


def _matches_suffix(name: str, suffixes: List[str]) -> bool:
    if not suffixes:
        return True
    lowered = name.lower()
    tokens = re.split(r"[_\-]+", lowered)
    for suffix in suffixes:
        if not suffix:
            continue
        if lowered.endswith(suffix) or lowered.endswith(f"_{suffix}") or suffix in tokens:
            return True
    return False


def _is_frame_file(path: Path) -> bool:
    """Return True for actual image frames; exclude hidden/metadata sidecars."""
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    return path.suffix.lower() in FRAME_EXTENSIONS


@dataclass
class VehicleStream:
    index: int
    label: str
    image_paths: List[Path]


@dataclass
class NegotiationOverlay:
    id: int
    frame_index: int
    hold_frames: int
    lines: List[str]
    color: Tuple[int, int, int]


def list_vehicle_streams(root: Path, stream_suffixes: Optional[List[str]] = None) -> List[VehicleStream]:
    """Discover vehicle-specific image folders within the given root."""
    streams: List[VehicleStream] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue

        m = VEHICLE_DIR_PATTERN.match(entry.name)
        if not m:
            continue
        if stream_suffixes and not _matches_suffix(entry.name, stream_suffixes):
            continue

        index = int(m.group(1))   # the (\d+) capture

        image_paths = sorted(
            (path for path in entry.iterdir() if _is_frame_file(path)),
            key=lambda p: p.stem,
        )
        if not image_paths:
            continue

        streams.append(
            VehicleStream(index=index, label=f"Vehicle {index}", image_paths=image_paths)
        )

    if not streams:
        raise FileNotFoundError(
            f"No vehicle folders with images were found in {root}. "
            "Expected folders like rgb_front_0, rgb_front_1 (or ego_vehicle_* if present)."
        )

    streams.sort(key=lambda s: s.index)
    return streams


def _scenario_has_vehicle_images(scenario_dir: Path) -> bool:
    """Return True if scenario_dir contains at least one ego_vehicle_* folder with images."""
    if not scenario_dir.is_dir():
        return False

    for child in scenario_dir.iterdir():
        if not (child.is_dir() and VEHICLE_DIR_PATTERN.match(child.name)):
            continue
        for item in child.iterdir():
            if _is_frame_file(item):
                return True
    return False


def _scenario_has_flat_images(scenario_dir: Path) -> bool:
    """Return True if scenario_dir contains image files directly (no vehicle subfolders required)."""
    if not scenario_dir.is_dir():
        return False
    try:
        for item in scenario_dir.iterdir():
            if _is_frame_file(item):
                return True
    except Exception:
        return False
    return False


def discover_scenario_dirs(root: Path) -> List[Path]:
    """Recursively find scenario directories underneath root."""
    scenarios: List[Path] = []
    seen: set[Path] = set()
    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        if candidate in seen:
            continue
        if _scenario_has_vehicle_images(candidate):
            scenarios.append(candidate)
            seen.add(candidate)
    scenarios.sort()
    return scenarios


def remove_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def chunk_word(word: str, font, font_scale: float, thickness: int, max_width: int) -> List[str]:
    """Split an oversized word into chunks that fit within max_width."""
    chunks: List[str] = []
    current = ""
    for char in word:
        tentative = current + char
        width = cv2.getTextSize(tentative, font, font_scale, thickness)[0][0]
        if width <= max_width or not current:
            current = tentative
        else:
            chunks.append(current)
            current = char
    if current:
        chunks.append(current)
    return chunks


def wrap_line(
    line: str, font, font_scale: float, thickness: int, max_width: int
) -> List[str]:
    """Wrap a line of text to fit within max_width."""
    words = line.split()
    if not words:
        return [""]

    wrapped: List[str] = []
    current = ""
    for word in words:
        tentative = word if not current else f"{current} {word}"
        width = cv2.getTextSize(tentative, font, font_scale, thickness)[0][0]
        if width <= max_width:
            current = tentative
            continue

        if current:
            wrapped.append(current)
            current = ""

        word_width = cv2.getTextSize(word, font, font_scale, thickness)[0][0]
        if word_width <= max_width:
            current = word
            continue

        chunks = chunk_word(word, font, font_scale, thickness, max_width)
        if chunks:
            wrapped.extend(chunks[:-1])
            current = chunks[-1]

    if current:
        wrapped.append(current)
    return wrapped


def discover_log_path(root: Path) -> Optional[Path]:
    """Return the first .log file found within the directory."""
    nested = root / "log"
    if nested.is_dir():
        candidate = nested / "log.log"
        if candidate.exists():
            return candidate

    for path in root.rglob("*.log"):
        return path
    return None


def _stable_color_for_event(event_id: int) -> Tuple[int, int, int]:
    """Pick a readable BGR color that is stable per event."""
    rng = np.random.default_rng(event_id + 2024)
    # Sample until the luminance is within a readable range.
    for _ in range(8):
        color = rng.integers(60, 230, size=3, dtype=np.int32)
        b, g, r = int(color[0]), int(color[1]), int(color[2])
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if 80 <= luminance <= 220:
            return (b, g, r)
    color = rng.integers(90, 210, size=3, dtype=np.int32)
    return (int(color[0]), int(color[1]), int(color[2]))


def parse_negotiation_events(
    log_path: Optional[Path],
    fps: float,
    min_hold_seconds: float,
    words_per_minute: float,
    reading_factor: float,
) -> List[NegotiationOverlay]:
    if not log_path or not log_path.exists():
        return []

    overlays: List[NegotiationOverlay] = []
    last_step_time = 0.0
    capturing = False
    negotiation_time = 0.0
    event_counter = 0

    current_actor: Optional[int] = None
    current_messages: List[Tuple[Optional[int], str]] = []
    current_action_list: Optional[str] = None
    current_short_analysis: Optional[str] = None

    words_per_second = max(1e-3, words_per_minute / 60.0)
    reading_factor = max(0.05, reading_factor)

    def finalize_event() -> None:
        nonlocal event_counter
        if (
            not current_messages
            and not current_action_list
            and not current_short_analysis
        ):
            return

        ordered_lines: List[str] = []
        for car_id, raw_text in current_messages:
            clean_text = (raw_text or "").strip()
            clean_text = re.sub(r"^ID\\s*\\d+\\s*:\\s*", "", clean_text, flags=re.IGNORECASE)
            actor_label = f"Vehicle {car_id}" if car_id is not None else "Global"
            line = f"{actor_label}: {clean_text if clean_text else '(no message)'}"
            ordered_lines.append(line)
        if current_action_list:
            ordered_lines.append(f"Action List: {current_action_list}")

        if current_short_analysis:
            ordered_lines.append(f"Short Analysis: {current_short_analysis}")

        if not ordered_lines:
            return

        word_count = sum(len(line.split()) for line in ordered_lines if line.strip())

        hold_seconds = max(min_hold_seconds, (word_count / words_per_second) * reading_factor)
        hold_frames = max(1, int(round(hold_seconds * fps)))
        frame_index = max(0, int(round(negotiation_time * fps)))
        color = _stable_color_for_event(event_counter)

        overlays.append(
            NegotiationOverlay(
                id=event_counter,
                frame_index=frame_index,
                hold_frames=hold_frames,
                lines=ordered_lines[:],
                color=color,
            )
        )
        event_counter += 1

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            clean_line = remove_ansi(raw_line.rstrip("\n"))
            step_match = STEP_TIME_PATTERN.search(clean_line)
            if step_match:
                try:
                    last_step_time = float(step_match.group(1))
                except ValueError:
                    pass

            if "Conflict exist, start negotiation" in clean_line:
                capturing = True
                negotiation_time = last_step_time
                current_actor = None
                current_messages.clear()
                current_action_list = None
                current_short_analysis = None
                continue

            if not capturing:
                continue

            car_match = re.search(r"car_id:\s*(\d+)", clean_line)
            if car_match:
                try:
                    current_actor = int(car_match.group(1))
                except ValueError:
                    current_actor = None
                continue

            stripped = clean_line.strip()
            if stripped.startswith("message:"):
                message_text = stripped.split("message:", 1)[1].strip()
                message_text = message_text.strip('" ')
                current_messages.append((current_actor, message_text))
                continue

            if stripped.startswith("action_list:"):
                raw_value = stripped.split("action_list:", 1)[1].strip()
                formatted = raw_value
                try:
                    parsed = ast.literal_eval(raw_value)
                    if isinstance(parsed, dict):
                        pieces: List[str] = []
                        for key in sorted(
                            parsed.keys(),
                            key=lambda x: int(x) if str(x).isdigit() else str(x),
                        ):
                            detail = parsed[key]
                            if isinstance(detail, dict):
                                nav = detail.get("nav")
                                speed = detail.get("speed")
                                parts = [part for part in (nav, speed) if part]
                                desc = " / ".join(parts) if parts else str(detail)
                            else:
                                desc = str(detail)
                            pieces.append(f"Vehicle {key}: {desc}")
                        if pieces:
                            formatted = "; ".join(pieces)
                except Exception:
                    formatted = raw_value
                current_action_list = formatted
                continue

            if stripped.startswith("Short analysis"):
                summary = stripped.split("Short analysis:", 1)[1].strip()
                current_short_analysis = summary
                continue

            lowered = stripped.lower()
            if (
                "comm_results" in lowered
                or "negotiation done" in lowered
                or lowered.startswith("vlm using negotiation results")
            ):
                finalize_event()
                capturing = False

    return overlays


def add_label(image: np.ndarray, label: str) -> Tuple[np.ndarray, int]:
    """Overlay a larger label in the top-left corner and return the bottom of the label box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    padding = LABEL_PADDING
    text_size, _ = cv2.getTextSize(label, font, scale, thickness)
    box_pt1 = (padding - 6, padding - 10)
    box_pt2 = (padding + text_size[0] + 6, padding + text_size[1])

    overlay = image.copy()
    cv2.rectangle(overlay, box_pt1, box_pt2, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, dst=image)
    text_origin = (padding, padding + text_size[1] - 4)
    cv2.putText(
        image,
        label,
        text_origin,
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    label_bottom = text_origin[1] + 6
    return image, label_bottom


def render_negotiation_slot(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    lines: List[str],
    color: Tuple[int, int, int],
) -> None:
    """Render a negotiation box anchored within the negotiation column."""
    base_color = (30, 30, 30)
    border_thickness = max(2, int(round(height * 0.01)))
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), base_color, -1)
    cv2.rectangle(
        canvas,
        (x, y),
        (x + width - 1, y + height - 1),
        (0, 0, 0),
        border_thickness,
    )

    cleaned = [line.strip() for line in lines if line.strip()]
    if not cleaned:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    margin_ratio = 0.05
    margin_x = max(6, int(round(width * margin_ratio)))
    margin_y = max(6, int(round(height * margin_ratio)))
    available_width = max(20, width - 2 * margin_x)
    available_height = max(20, height - 2 * margin_y)

    min_scale = 0.1
    max_scale = 6.0

    def layout(scale: float) -> Tuple[List[str], int, int, int, int, int]:
        """Return (wrapped_lines, thickness, line_height, spacing, max_width, total_height)."""
        thickness = max(1, int(round(max(scale * 0.85, 1.0))))
        baseline = cv2.getTextSize("Ag", font, scale, thickness)[0][1]
        spacing = max(4, int(round(baseline * 0.3)))
        wrapped: List[str] = []
        for raw_line in cleaned:
            segments = wrap_line(raw_line, font, scale, thickness, available_width)
            if not segments:
                continue
            wrapped.extend(segments)

        if not wrapped:
            wrapped = [cleaned[0]]

        max_width = 0
        for segment in wrapped:
            max_width = max(max_width, cv2.getTextSize(segment, font, scale, thickness)[0][0])

        total_height = len(wrapped) * baseline
        if len(wrapped) > 1:
            total_height += (len(wrapped) - 1) * spacing
        return wrapped, thickness, baseline, spacing, max_width, total_height

    def fits(scale: float) -> Tuple[bool, Tuple[List[str], int, int, int, int, int]]:
        layout_result = layout(scale)
        max_width, total_height = layout_result[4], layout_result[5]
        return (max_width <= available_width and total_height <= available_height), layout_result

    scale = min_scale
    layout_result = layout(scale)
    attempts = 0
    while (layout_result[4] > available_width or layout_result[5] > available_height) and attempts < 8:
        scale *= 0.7
        layout_result = layout(scale)
        attempts += 1

    best_scale = scale
    best_layout = layout_result

    high = max(max_scale, best_scale * 1.5)
    fits_high, layout_high = fits(high)
    while fits_high and high < 24.0:
        best_scale = high
        best_layout = layout_high
        high *= 1.5
        fits_high, layout_high = fits(high)

    low = best_scale
    if high < low:
        high = low

    for _ in range(30):
        if high - low <= 1e-3:
            break
        mid = (low + high) / 2
        fits_mid, layout_mid = fits(mid)
        if fits_mid:
            best_scale = mid
            best_layout = layout_mid
            low = mid
        else:
            high = mid

    for _ in range(5):
        trial = best_scale * 1.08
        fits_trial, layout_trial = fits(trial)
        if fits_trial:
            best_scale = trial
            best_layout = layout_trial
        else:
            break

    wrapped_lines, thickness, line_height, spacing, _, total_height = best_layout

    b, g, r = color
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance < 185:
        blend = 0.55
        r = int(r + (255 - r) * blend)
        g = int(g + (255 - g) * blend)
        b = int(b + (255 - b) * blend)
        color = (b, g, r)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), color, -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, dst=canvas)

    text_color = (0, 0, 0)

    used_height = total_height
    vertical_extra = max(0, available_height - used_height)
    cursor_y = y + margin_y + line_height + vertical_extra // 2

    for line in wrapped_lines:
        cursor_x = x + margin_x
        cv2.putText(
            canvas,
            line,
            (cursor_x, cursor_y),
            font,
            best_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        bold_prefix: Optional[str] = None
        vehicle_match = re.match(r"(Vehicle\s+\d+\s*:\s*)", line)
        if vehicle_match:
            bold_prefix = vehicle_match.group(1)
        else:
            number_match = re.match(r"(\d+\.\s)", line)
            if number_match:
                bold_prefix = number_match.group(1)
            else:
                lowered_line = line.lower()
                if lowered_line.startswith("action list:"):
                    bold_prefix = line[: len("Action List:")]
                elif lowered_line.startswith("short analysis:"):
                    bold_prefix = line[: len("Short Analysis:")]

        if bold_prefix:
            bold_thickness = max(thickness + 1, thickness * 2)
            cv2.putText(
                canvas,
                bold_prefix,
                (cursor_x, cursor_y),
                font,
                best_scale,
                text_color,
                bold_thickness,
                cv2.LINE_AA,
            )

        cursor_y += line_height + spacing


def build_video(
    output_path: Path,
    streams: List[VehicleStream],
    fps: float,
    resize_factor: int,
    negotiation_events: List[NegotiationOverlay],
    side_stream: Optional[VehicleStream] = None,
    stack_mode: str = "vertical",
) -> None:
    def resize_with_aspect(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize into a fixed canvas while preserving aspect ratio (letterbox)."""
        target_w, target_h = size
        src_h, src_w = image.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        off_x = (target_w - new_w) // 2
        off_y = (target_h - new_h) // 2
        canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
        return canvas

    first_frame = cv2.imread(str(streams[0].image_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Unable to read {streams[0].image_paths[0]}")

    base_height, base_width = first_frame.shape[:2]
    target_size = (max(1, base_width // resize_factor), max(1, base_height // resize_factor))
    if stack_mode not in ("vertical", "horizontal"):
        raise ValueError(f"Unsupported stack_mode: {stack_mode}")

    if stack_mode == "horizontal":
        combined_height = target_size[1]
        combined_width = target_size[0] * len(streams)
    else:
        combined_height = target_size[1] * len(streams)
        combined_width = target_size[0]

    if side_stream:
        combined_width += target_size[0]

    # If simple mode (no negotiation events), omit sidebar column entirely.
    simple_mode = (not negotiation_events)
    column_width = 0 if simple_mode else max(220, int(round(target_size[0] * 0.4)))
    total_width = combined_width + column_width

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (total_width, combined_height)
    )

    main_frames = max(len(stream.image_paths) for stream in streams)
    side_frames = len(side_stream.image_paths) if side_stream else 0
    max_frames = max(main_frames, side_frames)
    # Calculate frame mapping for each stream
    def get_frame_index(total_frames, output_idx, output_total):
        if total_frames <= 1:
            return 0
        return int(round(output_idx * (total_frames - 1) / (output_total - 1))) if output_total > 1 else 0


    event_map: Dict[int, List[NegotiationOverlay]] = defaultdict(list)
    for event in negotiation_events:
        event_map[event.frame_index].append(event)

    active_events: List[Dict[str, object]] = []

    for frame_idx in tqdm(range(max_frames), desc="Compositing frames"):
        starting_events = event_map.get(frame_idx, [])
        for event in starting_events:
            active_events.append({"event": event, "remaining": event.hold_frames})

        frames: List[np.ndarray] = []
        for stream in streams:
            image_index = get_frame_index(len(stream.image_paths), frame_idx, max_frames)
            image = cv2.imread(str(stream.image_paths[image_index]))
            if image is None:
                raise RuntimeError(f"Unable to read {stream.image_paths[image_index]}")
            image = resize_with_aspect(image, target_size)
            image, _ = add_label(image, stream.label)
            frames.append(image)

        if len(frames) == 1:
            combined = frames[0]
        else:
            combined = cv2.hconcat(frames) if stack_mode == "horizontal" else cv2.vconcat(frames)

        if side_stream:
            side_index = get_frame_index(len(side_stream.image_paths), frame_idx, max_frames)
            side_image = cv2.imread(str(side_stream.image_paths[side_index]))
            if side_image is None:
                raise RuntimeError(f"Unable to read {side_stream.image_paths[side_index]}")
            side_image = resize_with_aspect(side_image, (target_size[0], combined_height))
            side_image, _ = add_label(side_image, side_stream.label)
            combined = cv2.hconcat([combined, side_image])
        if not simple_mode:
            column = np.zeros((combined_height, column_width, 3), dtype=np.uint8)
            column[:] = (24, 24, 24)

            active_event: Optional[NegotiationOverlay] = (
                active_events[-1]["event"] if active_events else None  # type: ignore[index]
            )
            if active_event:
                render_negotiation_slot(
                    column,
                    0,
                    0,
                    column_width,
                    combined_height,
                    active_event.lines,
                    active_event.color,
                )
            else:
                border_thickness = max(2, int(round(combined_height * 0.01)))
                cv2.rectangle(
                    column,
                    (0, 0),
                    (column_width - 1, combined_height - 1),
                    (45, 45, 45),
                    -1,
                )
                cv2.rectangle(
                    column,
                    (0, 0),
                    (column_width - 1, combined_height - 1),
                    (0, 0, 0),
                    border_thickness,
                )

        final_frame = combined if simple_mode else cv2.hconcat([column, combined])
        writer.write(final_frame)

        for active in list(active_events):
            active["remaining"] = int(active["remaining"]) - 1  # type: ignore[index]
            if active["remaining"] <= 0:
                active_events.remove(active)

    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a composite video from vehicle image folders, annotating each feed "
            "and presenting negotiation transcripts in a sidebar aligned with the speaking vehicle."
        )
    )
    parser.add_argument(
        "input_dir",
        help="Path to the scenario directory containing ego_vehicle_* (or similar) folders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path. Defaults to <input_dir>/output_v3.mp4.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where generated videos are saved when processing multiple scenarios. "
            "Defaults to <input_dir>/videos when batching."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Output video frames per second (default: 5.0).",
    )
    parser.add_argument(
        "--resize-factor",
        type=int,
        default=2,
        help="Factor by which to downscale each feed before stacking (default: 2).",
    )
    parser.add_argument(
        "--hold-seconds",
        "--min-hold-seconds",
        dest="min_hold_seconds",
        type=float,
        default=1.2,
        help="Minimum duration (in seconds) to display negotiation transcripts (default: 1.2).",
    )
    parser.add_argument(
        "--words-per-minute",
        type=float,
        default=240.0,
        help="Reading speed used to time transcript overlays (default: 240 WPM).",
    )
    parser.add_argument(
        "--reading-factor",
        type=float,
        default=0.5,
        help="Multiplier applied to the computed display time (default: 0.5).",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Optional path to a log file. If omitted, the script searches within the input directory.",
    )
    parser.add_argument(
        "--only-suffix",
        default=None,
        help=(
            "Filter inputs by suffix token. In flat-image mode, only filenames whose stem matches "
            "the suffix are kept (e.g., --only-suffix cam1 keeps *_cam1.*). In scenario mode, only "
            "stream folders whose names match the suffix are included. Multiple suffixes can be "
            "comma- or space-separated."
        ),
    )
    parser.add_argument(
        "--side-by-side-dir",
        default=None,
        help=(
            "Optional directory of images to append as an extra side-by-side column. This folder is "
            "not filtered by --only-suffix and should contain flat image files (no vehicle subfolders)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory {root} does not exist.")

    # Support two modes:
    # 1) Scenario mode: vehicle subfolders like rgb_front_0 with images.
    # 2) Flat mode: input_dir itself contains image files; build a single-stream video.
    is_single_scenario = _scenario_has_vehicle_images(root)
    is_flat_images = _scenario_has_flat_images(root)
    if is_single_scenario or is_flat_images:
        scenario_dirs = [root]
    else:
        scenario_dirs = discover_scenario_dirs(root)
        if not scenario_dirs:
            raise FileNotFoundError(
                f"No scenario directories with vehicle images (rgb_*, meta_*, ego_vehicle_*) were found under {root}."
            )

    if len(scenario_dirs) > 1 and args.output:
        raise ValueError(
            "The --output argument is only supported when a single scenario directory is processed. "
            "Use --output-dir to specify the destination for batch generation."
        )

    if len(scenario_dirs) > 1 and args.log_path:
        raise ValueError(
            "A custom --log-path cannot be combined with batch processing. "
            "Logs are discovered per scenario automatically."
        )

    if args.output_dir:
        batch_output_dir = Path(args.output_dir).expanduser().resolve()
        batch_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if len(scenario_dirs) > 1:
            batch_output_dir = (root / "videos").resolve()
            batch_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            batch_output_dir = None

    print(f"Discovered {len(scenario_dirs)} scenario directory(ies) to process.")
    suffixes = _parse_suffix_filter(args.only_suffix)
    side_stream: Optional[VehicleStream] = None
    if args.side_by_side_dir:
        side_root = Path(args.side_by_side_dir).expanduser().resolve()
        if not side_root.exists():
            raise FileNotFoundError(f"Side-by-side directory {side_root} does not exist.")
        if not _scenario_has_flat_images(side_root):
            raise FileNotFoundError(
                f"No flat images found in side-by-side directory {side_root}."
            )
        side_images = sorted(
            [p for p in side_root.iterdir() if _is_frame_file(p)],
            key=lambda p: p.stem,
        )
        if not side_images:
            raise FileNotFoundError(f"No images found in side-by-side directory {side_root}.")
        side_stream = VehicleStream(index=0, label=side_root.name, image_paths=side_images)

    for scenario_dir in scenario_dirs:
        stack_mode = "vertical"
        if len(scenario_dirs) > 1:
            print(f"\nProcessing scenario: {scenario_dir}")

        if args.output:
            output_path = Path(args.output).expanduser().resolve()
        elif batch_output_dir:
            output_path = batch_output_dir / f"{scenario_dir.name}.mp4"
        else:
            output_path = scenario_dir / "output_v3.mp4"

        # In flat mode, build a single stream from images directly in the folder.
        if _scenario_has_flat_images(scenario_dir):
            all_images = sorted(
                [p for p in scenario_dir.iterdir() if _is_frame_file(p)],
                key=lambda p: p.stem,
            )
            if not all_images:
                raise FileNotFoundError(f"No images found in {scenario_dir}.")

            if suffixes:
                streams = []
                for idx, suffix in enumerate(suffixes):
                    suffix_images = [p for p in all_images if _matches_suffix(p.stem, [suffix])]
                    if not suffix_images:
                        print(f"[WARN] No images found for suffix '{suffix}' in {scenario_dir}.")
                        continue
                    streams.append(
                        VehicleStream(index=idx, label=suffix, image_paths=suffix_images)
                    )
                if not streams:
                    raise FileNotFoundError(
                        f"No images found in {scenario_dir} matching suffix filter {suffixes}."
                    )
                if len(streams) > 1:
                    stack_mode = "horizontal"
            else:
                streams = [VehicleStream(index=0, label=f"{scenario_dir.name}", image_paths=all_images)]
            negotiation_events: List[NegotiationOverlay] = []
        else:
            streams = list_vehicle_streams(scenario_dir, stream_suffixes=suffixes)
            if not streams:
                message = (
                    f"No vehicle streams found in {scenario_dir} matching suffix filter {suffixes}."
                )
                if len(scenario_dirs) > 1:
                    print(f"[WARN] {message} Skipping.")
                    continue
                raise FileNotFoundError(message)

            log_path = (
                Path(args.log_path).expanduser().resolve()
                if args.log_path
                else discover_log_path(scenario_dir)
            )
            if log_path and not log_path.exists():
                log_path = None

            negotiation_events = parse_negotiation_events(
                log_path,
                args.fps,
                args.min_hold_seconds,
                args.words_per_minute,
                args.reading_factor,
            )
            if not negotiation_events:
                print("No negotiation events found; proceeding without transcript overlays.")
            else:
                print(f"Configured {len(negotiation_events)} negotiation overlay(s).")

        build_video(
            output_path=output_path,
            streams=streams,
            fps=args.fps,
            resize_factor=args.resize_factor,
            negotiation_events=negotiation_events,
            side_stream=side_stream,
            stack_mode=stack_mode,
        )
        print(f"Wrote video to {output_path}")


if __name__ == "__main__":
    main()
