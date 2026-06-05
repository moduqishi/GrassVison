"""Usage statistics tracking — in-memory with periodic JSON persistence."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import STATS_DIR

STATS_FILE = STATS_DIR / "stats.json"
MAX_HISTORY = 200
FLUSH_INTERVAL = 30  # seconds


class StatsTracker:
    def __init__(self):
        self._lock = threading.Lock()

        self.total_calls: int = 0
        self.vision_calls: int = 0
        self.vision_success: int = 0
        self.vision_failures: int = 0

        self.total_vision_prompt_tokens: int = 0
        self.total_vision_completion_tokens: int = 0
        self.total_source_prompt_tokens: int = 0
        self.total_source_completion_tokens: int = 0

        self.call_history: list[dict] = []

        self._load()
        self._start_flush_timer()

    # ── Recording ───────────────────────────────────────────────

    def record_call(
        self,
        model: str,
        images: int = 0,
        stream: bool = False,
        elapsed: float = 0,
        vision_used: bool = False,
        vision_success: bool = False,
        vision_tokens: dict | None = None,
        source_tokens: dict | None = None,
    ):
        with self._lock:
            self.total_calls += 1
            if vision_used:
                self.vision_calls += 1
                if vision_success:
                    self.vision_success += 1
                else:
                    self.vision_failures += 1

            if vision_tokens:
                self.total_vision_prompt_tokens += vision_tokens.get("prompt_tokens", 0)
                self.total_vision_completion_tokens += vision_tokens.get("completion_tokens", 0)
            if source_tokens:
                self.total_source_prompt_tokens += source_tokens.get("prompt_tokens", 0)
                self.total_source_completion_tokens += source_tokens.get("completion_tokens", 0)

            entry = {
                "time": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "images": images,
                "stream": stream,
                "elapsed": round(elapsed, 3),
                "vision_used": vision_used,
                "vision_success": vision_success,
            }
            if vision_tokens:
                entry["vision_tokens"] = vision_tokens
            if source_tokens:
                entry["source_tokens"] = source_tokens
            if vision_tokens or source_tokens:
                entry["total_tokens"] = (
                    (vision_tokens or {}).get("total_tokens", 0) +
                    (source_tokens or {}).get("total_tokens", 0)
                )

            self.call_history.insert(0, entry)  # newest first
            if len(self.call_history) > MAX_HISTORY:
                self.call_history = self.call_history[:MAX_HISTORY]

    # ── Query ───────────────────────────────────────────────────

    def summary(self) -> dict:
        with self._lock:
            total_vision_tokens = self.total_vision_prompt_tokens + self.total_vision_completion_tokens
            total_source_tokens = self.total_source_prompt_tokens + self.total_source_completion_tokens
            return {
                "total_calls": self.total_calls,
                "vision_calls": self.vision_calls,
                "vision_success": self.vision_success,
                "vision_failures": self.vision_failures,
                "total_vision_tokens": total_vision_tokens,
                "total_vision_prompt_tokens": self.total_vision_prompt_tokens,
                "total_vision_completion_tokens": self.total_vision_completion_tokens,
                "total_source_tokens": total_source_tokens,
                "total_source_prompt_tokens": self.total_source_prompt_tokens,
                "total_source_completion_tokens": self.total_source_completion_tokens,
                "total_all_tokens": total_vision_tokens + total_source_tokens,
            }

    def recent_calls(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self.call_history[:limit])

    def reset(self):
        with self._lock:
            self.total_calls = 0
            self.vision_calls = 0
            self.vision_success = 0
            self.vision_failures = 0
            self.total_vision_prompt_tokens = 0
            self.total_vision_completion_tokens = 0
            self.total_source_prompt_tokens = 0
            self.total_source_completion_tokens = 0
            self.call_history = []

    # ── Persistence ─────────────────────────────────────────────

    def _load(self):
        if STATS_FILE.exists():
            try:
                with open(STATS_FILE, "r") as f:
                    data = json.load(f)
                self.total_calls = data.get("total_calls", 0)
                self.vision_calls = data.get("vision_calls", 0)
                self.vision_success = data.get("vision_success", 0)
                self.vision_failures = data.get("vision_failures", 0)
                self.total_vision_prompt_tokens = data.get("total_vision_prompt_tokens", 0)
                self.total_vision_completion_tokens = data.get("total_vision_completion_tokens", 0)
                self.total_source_prompt_tokens = data.get("total_source_prompt_tokens", 0)
                self.total_source_completion_tokens = data.get("total_source_completion_tokens", 0)
                self.call_history = data.get("call_history", [])[:MAX_HISTORY]
            except Exception:
                pass

    def _flush(self):
        with self._lock:
            data = {
                "total_calls": self.total_calls,
                "vision_calls": self.vision_calls,
                "vision_success": self.vision_success,
                "vision_failures": self.vision_failures,
                "total_vision_prompt_tokens": self.total_vision_prompt_tokens,
                "total_vision_completion_tokens": self.total_vision_completion_tokens,
                "total_source_prompt_tokens": self.total_source_prompt_tokens,
                "total_source_completion_tokens": self.total_source_completion_tokens,
                "call_history": self.call_history[:MAX_HISTORY],
            }
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(STATS_FILE)

    def _start_flush_timer(self):
        def _loop():
            while True:
                time.sleep(FLUSH_INTERVAL)
                try:
                    self._flush()
                except Exception:
                    pass
        t = threading.Thread(target=_loop, daemon=True)
        t.start()


# Global singleton
_stats_tracker = StatsTracker()


def get_stats() -> StatsTracker:
    return _stats_tracker


def flush_stats():
    _stats_tracker._flush()
