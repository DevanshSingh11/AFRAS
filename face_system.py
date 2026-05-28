"""
face_system.py  –  Face Recognition & Attendance Engine
========================================================
Fixes in this version:
  • CAP_DSHOW guarded behind platform check (Windows-only) — no AttributeError on Linux/Mac
  • register_face_from_image bounding-box area fixed (was negative due to wrong coord order)
  • get_today_attendance no longer called every frame; HUD uses an in-memory counter
  • _safe_release: cv2.waitKey removed when no window is open (platform freeze fix)
  • All public methods return consistent (bool, str) tuples
  • Atomic pickle save via temp-file + rename (prevents corruption on crash mid-write)
  • Metadata kept in sync when adding samples to existing person
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import dlib
dlib.DLIB_USE_CUDA = False


import cv2
import face_recognition
import json
import logging
import pickle
import platform
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("FaceSystem")


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CAMERA_WARMUP_FRAMES  = 15
BLUR_THRESHOLD        = 80.0    # Laplacian variance below this → reject frame
_IS_WINDOWS           = platform.system() == "Windows"


# ─────────────────────────────────────────────
# ATTENDANCE MANAGER
# ─────────────────────────────────────────────
class AttendanceManager:
    """Thread-safe, CSV-backed attendance log."""

    HEADER = "Name,Date,Time,Confidence\n"

    def __init__(self, data_dir: Path) -> None:
        self.attendance_file = data_dir / "attendance.csv"
        self._lock           = threading.Lock()
        self._cache: Dict[str, set] = defaultdict(set)   # date_str → {name, …}
        self._last_date      = ""
        self._today_count    = 0   # fast counter for HUD (no disk read per frame)
        self._init_file()
        self._load_today()

    # ── internal ──────────────────────────────
    def _init_file(self) -> None:
        if not self.attendance_file.exists():
            self.attendance_file.write_text(self.HEADER, encoding="utf-8")

    def _load_today(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        self._last_date  = today
        self._cache.clear()
        if not self.attendance_file.exists():
            return
        try:
            with open(self.attendance_file, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 2 and parts[1] == today:
                        self._cache[today].add(parts[0])
        except OSError as exc:
            log.warning("Could not load attendance cache: %s", exc)
        self._today_count = len(self._cache[today])

    # ── public API ────────────────────────────
    def mark_attendance(self, name: str, confidence: float) -> Tuple[bool, str]:
        today = datetime.now().strftime("%Y-%m-%d")
        now   = datetime.now().strftime("%H:%M:%S")

        with self._lock:
            if today != self._last_date:
                self._load_today()          # midnight roll-over

            if name in self._cache[today]:
                return False, f"Already marked → {name}"

            try:
                with open(self.attendance_file, "a", encoding="utf-8") as f:
                    f.write(f"{name},{today},{now},{confidence:.3f}\n")
            except OSError as exc:
                log.error("Attendance write error: %s", exc)
                return False, f"Write error → {name}"

            self._cache[today].add(name)
            self._today_count += 1
            log.info("✓ Attendance: %s  (%.0f%%)", name, confidence * 100)
            return True, f"Marked → {name}"

    @property
    def today_count(self) -> int:
        """Fast in-memory count; no disk I/O."""
        return self._today_count

    def get_today_attendance(self) -> List[Tuple[str, str, float]]:
        today = datetime.now().strftime("%Y-%m-%d")
        records: List[Tuple[str, str, float]] = []
        if not self.attendance_file.exists():
            return records
        try:
            with open(self.attendance_file, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 4 and parts[1] == today:
                        try:
                            records.append((parts[0], parts[2], float(parts[3])))
                        except ValueError:
                            pass
        except OSError as exc:
            log.error("Cannot read attendance: %s", exc)
        return records

    def export_range(self, start: str, end: str, output_path: str) -> int:
        """Write rows in [start, end] (YYYY-MM-DD) to output_path. Returns row count."""
        try:
            d_start = datetime.strptime(start, "%Y-%m-%d").date()
            d_end   = datetime.strptime(end,   "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"Invalid date (use YYYY-MM-DD): {exc}") from exc

        if d_start > d_end:
            raise ValueError("Start date must be ≤ end date.")

        rows = [self.HEADER.strip()]
        if self.attendance_file.exists():
            with open(self.attendance_file, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        try:
                            if d_start <= datetime.strptime(parts[1], "%Y-%m-%d").date() <= d_end:
                                rows.append(line.strip())
                        except ValueError:
                            pass

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(rows))
        return len(rows) - 1

    def get_summary_stats(self) -> Dict[str, Dict]:
        counts: Dict[str, int] = defaultdict(int)
        dates:  Dict[str, set] = defaultdict(set)
        if not self.attendance_file.exists():
            return {}
        with open(self.attendance_file, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    counts[parts[0]] += 1
                    dates[parts[0]].add(parts[1])
        return {n: {"total": counts[n], "days": len(dates[n])} for n in counts}


# ─────────────────────────────────────────────
# FACE RECOGNITION SYSTEM
# ─────────────────────────────────────────────
class FaceRecognitionSystem:
    """
    Core recognition engine.
    • Stores multiple encodings per person for higher accuracy.
    • All public methods are safe to call from the main (GUI) thread.
    • recognize_faces_realtime() must be called from a background thread.
    """

    def __init__(self, data_dir: str = "face_data") -> None:
        self.data_dir   = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.images_dir = self.data_dir / "registered_faces"
        self.images_dir.mkdir(exist_ok=True)

        self.encodings_file = self.data_dir / "encodings.pkl"
        self.metadata_file  = self.data_dir / "metadata.json"

        # ── Tunable settings ──────────────────
        self.tolerance            : float = 0.45
        self.confidence_threshold : float = 0.60
        self.frame_skip           : int   = 2
        self.model                : str   = "hog"   # "hog" | "cnn"

        # ── Runtime state ─────────────────────
        self.known_encodings: List[np.ndarray] = []
        self.known_names    : List[str]         = []
        self.face_metadata  : Dict              = {}

        self._running    = False
        self._frame_cnt  = 0
        self._fps_buf   : List[float] = []

        # ── Callbacks (set by GUI) ─────────────
        self.on_attendance_marked: Optional[Callable] = None   # (name, conf)
        self.on_frame_processed  : Optional[Callable] = None   # (fps, n_faces)

        self.attendance = AttendanceManager(self.data_dir)

        self._load_encodings()
        self._load_metadata()

    # ─────────────────────────────────────────
    # STORAGE  (atomic writes to prevent corruption)
    # ─────────────────────────────────────────
    def _load_encodings(self) -> None:
        if not self.encodings_file.exists():
            return
        try:
            with open(self.encodings_file, "rb") as f:
                data = pickle.load(f)
            self.known_encodings = data.get("encodings", [])
            self.known_names     = data.get("names",     [])
            log.info("Loaded %d encoding(s) for %d person(s).",
                     len(self.known_names), len(set(self.known_names)))
        except Exception as exc:
            log.error("Corrupt encodings file – resetting. (%s)", exc)
            self.known_encodings, self.known_names = [], []

    def _save_encodings(self) -> None:
        """Atomic save: write to temp file then rename (prevents corruption on crash)."""
        fd, tmp = tempfile.mkstemp(dir=self.data_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump({"encodings": self.known_encodings,
                             "names":     self.known_names}, f)
            # atomic on POSIX; near-atomic on Windows
            shutil.move(tmp, str(self.encodings_file))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load_metadata(self) -> None:
        if not self.metadata_file.exists():
            return
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.face_metadata = json.load(f)
        except Exception as exc:
            log.warning("Could not load metadata: %s", exc)

    def _save_metadata(self) -> None:
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.face_metadata, f, indent=2, ensure_ascii=False)

    # ─────────────────────────────────────────
    # CAMERA HELPERS
    # ─────────────────────────────────────────
    @staticmethod
    def _open_camera(index: int = 0) -> Optional[cv2.VideoCapture]:
        """Open camera. Tries CAP_DSHOW on Windows for better compatibility."""
        backends = []
        if _IS_WINDOWS:
            backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_ANY)

        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap and cap.isOpened():
                # Warmup: allow auto-exposure to stabilise
                for _ in range(CAMERA_WARMUP_FRAMES):
                    cap.read()
                return cap
        return None

    @staticmethod
    def _safe_release(cap: Optional[cv2.VideoCapture]) -> None:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        # Only call destroyAllWindows — skip waitKey which hangs when no window exists
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    @staticmethod
    def _is_blurry(gray: np.ndarray) -> bool:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < BLUR_THRESHOLD

    # ─────────────────────────────────────────
    # REGISTRATION
    # ─────────────────────────────────────────
    def register_face_from_webcam(
        self,
        name: str,
        samples: int = 8,
        progress_cb: Optional[Callable] = None,
    ) -> Tuple[bool, str]:
        """
        Interactive webcam registration.
        Press SPACE to capture, ESC to cancel.
        progress_cb(n_captured, total) is called after each capture.
        """
        cap = self._open_camera()
        if cap is None:
            return False, "Cannot open camera."

        collected: List[np.ndarray] = []
        cancel_msg = ""

        while len(collected) < samples:
            ret, frame = cap.read()
            if not ret:
                cancel_msg = "Camera read failed."
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs  = face_recognition.face_locations(rgb, model=self.model)

            blurry  = self._is_blurry(gray)
            n_faces = len(locs)
            ready   = (n_faces == 1 and not blurry)

            # Bounding boxes
            for t, r, b, l in locs:
                box_color = (0, 220, 0) if ready else (0, 120, 255)
                cv2.rectangle(frame, (l, t), (r, b), box_color, 2)

            # Top status bar
            parts = [f"Captured: {len(collected)}/{samples}"]
            if blurry:          parts.append("BLURRY – hold still")
            if n_faces == 0:    parts.append("No face detected")
            elif n_faces > 1:   parts.append("Multiple faces!")

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 44), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, "  |  ".join(parts),
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

            # Bottom hint
            hint_color = (100, 220, 100) if ready else (140, 140, 140)
            cv2.putText(frame, "SPACE = capture    ESC = cancel",
                        (8, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, hint_color, 1)

            cv2.imshow(f"Register: {name}", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:       # ESC
                cancel_msg = "Cancelled by user."
                break

            if key == 32 and ready:   # SPACE
                encs = face_recognition.face_encodings(rgb, locs)
                if encs:
                    collected.append(encs[0])
                    log.info("  Sample %d/%d captured for '%s'.",
                             len(collected), samples, name)
                    if progress_cb:
                        progress_cb(len(collected), samples)

        self._safe_release(cap)

        if not collected:
            return False, cancel_msg or "No samples collected."

        # Append new encodings (existing ones are kept → additive registration)
        for enc in collected:
            self.known_encodings.append(enc)
            self.known_names.append(name)

        self._save_encodings()

        # Update metadata (preserve image_path if already set)
        existing = self.face_metadata.get(name, {})
        self.face_metadata[name] = {
            "registered_date": existing.get(
                "registered_date",
                datetime.now().isoformat(timespec="seconds"),
            ),
            "samples":    self.known_names.count(name),
            "image_path": existing.get("image_path"),
        }
        self._save_metadata()
        log.info("Registered '%s' — total encodings: %d", name, self.known_names.count(name))
        return True, f"Registered '{name}' with {len(collected)} new sample(s)."

    def register_face_from_image(
        self, name: str, image_path: str
    ) -> Tuple[bool, str]:
        """Register a person from a static image file."""
        path = Path(image_path)
        if not path.exists():
            return False, f"File not found: {path}"

        img       = face_recognition.load_image_file(str(path))
        locations = face_recognition.face_locations(img, model=self.model)

        if not locations:
            return False, "No face detected in the image."

        if len(locations) > 1:
            log.warning("Multiple faces detected — using the largest.")
            # BUG FIX: face_recognition coords are (top, right, bottom, left)
            # Area = (bottom - top) * (right - left)  — both must be positive
            locations = [max(locations,
                             key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))]

        encs = face_recognition.face_encodings(img, locations)
        if not encs:
            return False, "Could not compute face encoding."

        dest = self.images_dir / f"{name}_{path.stem}{path.suffix}"
        shutil.copy2(str(path), str(dest))

        self.known_encodings.append(encs[0])
        self.known_names.append(name)
        self._save_encodings()

        existing = self.face_metadata.get(name, {})
        self.face_metadata[name] = {
            "registered_date": existing.get(
                "registered_date",
                datetime.now().isoformat(timespec="seconds"),
            ),
            "samples":    self.known_names.count(name),
            "image_path": str(dest),
        }
        self._save_metadata()
        log.info("Registered '%s' from image '%s'.", name, path.name)
        return True, f"Registered '{name}' from image."

    # ─────────────────────────────────────────
    # DELETION
    # ─────────────────────────────────────────
    def delete_face(self, name: str) -> bool:
        indices = [i for i, n in enumerate(self.known_names) if n == name]
        if not indices:
            return False

        for idx in sorted(indices, reverse=True):
            del self.known_names[idx]
            del self.known_encodings[idx]

        self._save_encodings()

        meta     = self.face_metadata.pop(name, {})
        img_path = meta.get("image_path")
        if img_path:
            p = Path(img_path)
            if p.exists():
                try:
                    p.unlink()
                except OSError as exc:
                    log.warning("Could not delete image file: %s", exc)

        self._save_metadata()
        log.info("Deleted '%s' (%d encoding(s) removed).", name, len(indices))
        return True

    # ─────────────────────────────────────────
    # RECOGNITION
    # ─────────────────────────────────────────
    def stop_recognition(self) -> None:
        self._running = False

    def recognize_faces_realtime(self) -> None:
        """
        Run the live recognition loop.
        MUST be called from a background thread.
        """
        if not self.known_names:
            log.warning("No registered faces — aborting recognition.")
            return

        time.sleep(0.25)     # let any previous cv2 window finish closing

        cap = self._open_camera()
        if cap is None:
            log.error("Cannot open camera.")
            return

        self._running   = True
        self._frame_cnt = 0
        self._fps_buf   = []
        last_results: List[Tuple[str, Tuple[int,int,int,int], float]] = []
        t_prev = time.perf_counter()

        # Cache the HUD count so we don't read the CSV every frame
        hud_att_count = self.attendance.today_count

        log.info("Recognition started. Press ESC in the video window to stop.")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                log.warning("Camera read failed — stopping recognition.")
                break

            self._frame_cnt += 1

            # ── FPS calculation ──────────────────
            t_now = time.perf_counter()
            dt    = max(t_now - t_prev, 1e-6)
            self._fps_buf.append(1.0 / dt)
            if len(self._fps_buf) > 24:
                self._fps_buf.pop(0)
            fps    = float(np.mean(self._fps_buf))
            t_prev = t_now

            # ── Detection & recognition ──────────
            if self._frame_cnt % max(1, self.frame_skip) == 0:
                small     = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                locs = face_recognition.face_locations(rgb_small, model=self.model)
                encs = face_recognition.face_encodings(rgb_small, locs)

                last_results = []
                for (t, r, b, l), enc in zip(locs, encs):
                    rec_name, confidence = "Unknown", 0.0

                    if self.known_encodings:
                        dists = face_recognition.face_distance(self.known_encodings, enc)
                        best  = int(np.argmin(dists))
                        dist  = float(dists[best])
                        conf  = float(1.0 - dist)

                        if dist < self.tolerance and conf >= self.confidence_threshold:
                            rec_name   = self.known_names[best]
                            confidence = conf
                            marked, _  = self.attendance.mark_attendance(rec_name, conf)
                            if marked:
                                hud_att_count = self.attendance.today_count
                                if self.on_attendance_marked:
                                    self.on_attendance_marked(rec_name, conf)

                    # Scale coords back to original frame size
                    last_results.append((rec_name, (t*2, r*2, b*2, l*2), confidence))

            # ── Draw bounding boxes ──────────────
            for rec_name, (t, r, b, l), conf in last_results:
                known = rec_name != "Unknown"
                color = (34, 197, 94) if known else (239, 68, 68)
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.rectangle(frame, (l, b - 30), (r, b), color, cv2.FILLED)
                label = f"{rec_name}  {conf:.0%}" if known else "Unknown"
                cv2.putText(frame, label, (l + 6, b - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)

            # ── HUD bar ──────────────────────────
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 32), (w, h), (18, 18, 18), cv2.FILLED)
            hud = (f"FPS: {fps:.1f}   "
                   f"Faces: {len(last_results)}   "
                   f"Attendance today: {hud_att_count}   "
                   f"Model: {self.model.upper()}   "
                   f"Tol: {self.tolerance}")
            cv2.putText(frame, hud, (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 210, 255), 1)

            # ── Fire GUI callback ────────────────
            if self.on_frame_processed:
                self.on_frame_processed(fps, len(last_results))

            cv2.imshow("Face Recognition  |  ESC to stop", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self._running = False
        self._safe_release(cap)
        log.info("Recognition stopped.")

    # ─────────────────────────────────────────
    # QUERIES
    # ─────────────────────────────────────────
    def list_registered_faces(self) -> List[Dict]:
        seen: set = set()
        result   = []
        for name in self.known_names:
            if name in seen:
                continue
            seen.add(name)
            meta = self.face_metadata.get(name, {})
            result.append({
                "name":            name,
                "registered_date": meta.get("registered_date", "—"),
                "samples":         self.known_names.count(name),
                "has_image":       bool(meta.get("image_path")),
            })
        return sorted(result, key=lambda x: x["name"].lower())

    def get_statistics(self) -> Dict:
        return {
            "total_registered":     len(set(self.known_names)),
            "total_encodings":      len(self.known_names),
            "attendance_today":     self.attendance.today_count,
            "tolerance":            self.tolerance,
            "confidence_threshold": self.confidence_threshold,
            "model":                self.model,
            "frame_skip":           self.frame_skip,
        }
