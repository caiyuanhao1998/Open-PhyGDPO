"""Count completed videos using anonymous markers on local disk."""

from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import re
from threading import Lock
import time
from zoneinfo import ZoneInfo


TIMEZONE = ZoneInfo("America/Los_Angeles")
EVENT_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})_[a-f0-9]{32}\.txt")
LOG = logging.getLogger(__name__)


def event_name(video_id, completed):
    return f"{completed.astimezone(TIMEZONE).date().isoformat()}_{video_id}.txt"


class UsageStats:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._lock = Lock()
        self._cached = None
        self._cache_day = None
        self._read_at = 0

    def record(self, video_id):
        completed = datetime.now(timezone.utc)
        name = event_name(video_id, completed)
        if not EVENT_NAME.fullmatch(name):
            raise ValueError("Expected a completed video ID")

        with self._lock:
            for delay in (0, 0.15, 0.45):
                if delay:
                    time.sleep(delay)
                try:
                    with (self.directory / name).open("x", encoding="ascii") as marker:
                        marker.write(completed.isoformat() + "\n")
                        marker.flush()
                        os.fsync(marker.fileno())
                    break
                except FileExistsError:
                    break
                except OSError as error:
                    last_error = error
            else:
                LOG.warning("SUCCESS_COUNT_NOT_SAVED event=%s error=%s", name, type(last_error).__name__)
            self._cached = None

    def counts(self, *, now=None):
        today = (now or datetime.now(timezone.utc)).astimezone(TIMEZONE).date()
        end = today.isoformat()
        start = (today - timedelta(days=29)).isoformat()
        with self._lock:
            if self._cached is not None and self._cache_day == today and time.monotonic() - self._read_at < 30:
                return self._cached
            last_30_days = today_count = 0
            try:
                with os.scandir(self.directory) as entries:
                    for entry in entries:
                        match = EVENT_NAME.fullmatch(entry.name)
                        if match and start <= match[1] <= end and entry.is_file():
                            last_30_days += 1
                            today_count += match[1] == end
            except OSError as error:
                LOG.warning("Success counts unavailable (%s)", type(error).__name__)
                return None
            self._cached = last_30_days, today_count
            self._cache_day = today
            self._read_at = time.monotonic()
            return self._cached
