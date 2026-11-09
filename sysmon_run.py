"""Read-only run paths and timezone-independent time windows shared by QA tools."""
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def parse_timestamp(value, require_timezone=False):
    """Accept epoch seconds or ISO 8601; CLI times without an offset mean UTC."""
    text = str(value).strip()
    try:
        result = float(text)
    except ValueError:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            if require_timezone:
                raise ValueError("Metadata timestamps must include a timezone offset")
            parsed = parsed.replace(tzinfo=timezone.utc)
        result = parsed.timestamp()
    if not math.isfinite(result):
        raise ValueError("Timestamp must be finite")
    datetime.fromtimestamp(result, timezone.utc)  # Reject unsupported dates early.
    return result


def iso_utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def open_readonly(db_path):
    db = Path(db_path).expanduser().resolve()
    if not db.is_file():
        raise FileNotFoundError(f"Database does not exist: {db}")
    return sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, check_same_thread=False)


def resolve_inputs(run_dir=None, db_path=None):
    run = Path(run_dir).expanduser().resolve() if run_dir else None
    if not run and db_path:
        candidate = Path(db_path).expanduser().resolve().parent
        if candidate.name == "sysmon" and candidate.parent.name == "metrics":
            candidate = candidate.parent.parent
            if (candidate / "metadata").is_dir():
                run = candidate
    if db_path:
        db = Path(db_path).expanduser().resolve()
    elif run:
        db = run / "metrics" / "sysmon" / "metrics.db"
    else:
        raise ValueError("Provide --run-dir or --db")
    if not db.is_file():
        raise FileNotFoundError(f"Database does not exist: {db}")
    return run, db


def resolve_window(run_dir=None, start=None, end=None, buffer_minutes=0):
    if not math.isfinite(buffer_minutes) or buffer_minutes < 0:
        raise ValueError("--buffer-minutes must be finite and nonnegative")
    if (start is None) != (end is None):
        raise ValueError("--from and --to must be provided together")
    if start is not None:
        t_from, t_to = parse_timestamp(start), parse_timestamp(end)
        source = "explicit"
    elif run_dir:
        meta = Path(run_dir) / "metadata"
        t_from = parse_timestamp((meta / "start-time.txt").read_text().strip(), True)
        t_to = parse_timestamp((meta / "end-time.txt").read_text().strip(), True)
        source = "run_metadata"
    else:
        return None
    if t_to <= t_from:
        raise ValueError("End time must be later than start time")
    return {
        "source": source,
        "start_epoch": t_from - buffer_minutes * 60,
        "end_epoch": t_to + buffer_minutes * 60,
        "start_utc": iso_utc(t_from - buffer_minutes * 60),
        "end_utc": iso_utc(t_to + buffer_minutes * 60),
        "unbuffered_start_utc": iso_utc(t_from),
        "unbuffered_end_utc": iso_utc(t_to),
        "buffer_minutes": buffer_minutes,
    }
