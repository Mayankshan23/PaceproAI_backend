"""
PacePro AI - Temporary Video Session Manager
============================================
Manages temporary annotated skeleton video storage and streaming.
CRITICAL PRIVACY & PERFORMANCE RULE:
- Videos are stored in a temporary session directory ONLY.
- Videos are NEVER persisted to MongoDB.
- Sessions older than 30 minutes are automatically pruned.
"""

import os
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict

TEMP_BASE_DIR = Path(__file__).resolve().parent / "temp_annotated"
TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True)


class TempVideoManager:
    """Manages temporary disk-backed video sessions with automated time-based eviction."""

    def __init__(self, base_dir: Path = TEMP_BASE_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> str:
        """Creates a unique temporary session ID."""
        self.cleanup_expired_sessions()
        session_id = f"sess_{uuid.uuid4().hex[:10]}"
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_id

    def get_session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def get_output_path(self, session_id: str, view_name: str) -> str:
        clean_view = "".join(c for c in view_name if c.isalnum() or c in "_-")
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir / f"{clean_view}_annotated.mp4")

    def get_video_file(self, session_id: str, view_name: str) -> Optional[Path]:
        clean_view = "".join(c for c in view_name if c.isalnum() or c in "_-")
        target = self.base_dir / session_id / f"{clean_view}_annotated.mp4"
        if target.exists() and target.is_file():
            return target
        return None

    def cleanup_expired_sessions(self, max_age_seconds: int = 1800):
        """Deletes any temporary video session folder older than max_age_seconds (30 mins)."""
        now = time.time()
        try:
            for item in self.base_dir.iterdir():
                if item.is_dir():
                    age = now - item.stat().st_mtime
                    if age > max_age_seconds:
                        try:
                            shutil.rmtree(item, ignore_errors=True)
                        except Exception:
                            pass
        except Exception:
            pass


temp_video_manager = TempVideoManager()
