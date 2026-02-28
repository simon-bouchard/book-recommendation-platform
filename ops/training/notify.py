# ops/training/notify.py
"""
Notification helpers for the automated training pipeline.

Supports three independent channels, each enabled only when its environment
variables are set:

  - Email (SMTP):   NOTIFY_EMAIL_TO, NOTIFY_SMTP_HOST, NOTIFY_SMTP_USER,
                    NOTIFY_SMTP_PASSWORD, and optionally NOTIFY_EMAIL_FROM,
                    NOTIFY_SMTP_PORT (default 587), NOTIFY_SMTP_TLS (default true).
  - Slack webhook:  NOTIFY_SLACK_WEBHOOK
  - Healthchecks:   HEALTHCHECKS_STEP_START / _SUCCESS / _FAIL

All channels fail silently so a notification outage never blocks the pipeline.
Set NOTIFY_STEP_MUTE=start,ok to suppress everything except failures.
"""

import os
import smtplib
import time
from contextlib import contextmanager
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode

import requests

# ---------------------------------------------------------------------------
# Channel configuration (read once at import time)
# ---------------------------------------------------------------------------

# Email
_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO", "").strip()
_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM", _EMAIL_TO).strip()
_SMTP_HOST = os.getenv("NOTIFY_SMTP_HOST", "").strip()
_SMTP_PORT = int(os.getenv("NOTIFY_SMTP_PORT", "587"))
_SMTP_USER = os.getenv("NOTIFY_SMTP_USER", "").strip()
_SMTP_PASSWORD = os.getenv("NOTIFY_SMTP_PASSWORD", "").strip()
_SMTP_TLS = os.getenv("NOTIFY_SMTP_TLS", "true").strip().lower() != "false"

# Slack
_SLACK = os.getenv("NOTIFY_SLACK_WEBHOOK", "").strip()

# Healthchecks
_HC_START = os.getenv("HEALTHCHECKS_STEP_START", "").strip()
_HC_SUCCESS = os.getenv("HEALTHCHECKS_STEP_SUCCESS", "").strip()
_HC_FAIL = os.getenv("HEALTHCHECKS_STEP_FAIL", "").strip()

DEFAULT_MUTE: set = {
    s.strip().lower() for s in os.getenv("NOTIFY_STEP_MUTE", "").split(",") if s.strip()
}

# Status colours used by Slack
_STATUS_COLORS = {"start": "#888888", "ok": "#2EB67D", "fail": "#E01E5A"}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def notify(status: str, title: str, body: str = "", log_tail: Optional[str] = None) -> None:
    """
    Dispatch a notification to all configured channels.

    Args:
        status:   One of 'start', 'ok', or 'fail'.
        title:    Short summary line shown as the message header/subject.
        body:     Optional detail text.
        log_tail: Optional trailing log lines appended after body (email and Slack).
    """
    _notify_email(status, title, body, log_tail)
    _notify_slack(status, title, body, log_tail)
    _notify_healthchecks(status, body)


def read_tail(path: Optional[str], n: int = 60) -> Optional[str]:
    """
    Return the last n lines of a text file, or None if the file is unreadable.

    Reads from the end of the file in blocks to avoid loading large log files
    into memory.
    """
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                r = min(block, size)
                size -= r
                f.seek(size)
                data = f.read(r) + data
        return "\n".join(data.decode("utf-8", "replace").splitlines()[-n:])
    except Exception:
        return None


# messages spec can be a plain string or a callable that receives the step context
MessageSpec = Optional[Callable[[Dict[str, Any]], str] | str]


@contextmanager
def notify_step(
    title: str,
    log_path: Optional[str] = None,
    messages: Optional[Dict[str, MessageSpec]] = None,
    mute: Optional[set | list] = None,
):
    """
    Context manager that sends start, success, and failure notifications.

    Usage::

        with notify_step("Model training", log_path="/tmp/train.log"):
            train()

    Args:
        title:    Human-readable name of the step, used in all notifications.
        log_path: Path to a log file whose tail is appended on failure.
        messages: Optional dict with keys 'start', 'ok', 'fail'. Each value
                  may be a plain string or a callable receiving the step context
                  dict (keys: title, log_path, started_at, duration_s,
                  exit_code, exc).
        mute:     Iterable of statuses to suppress, e.g. {'start', 'ok'} for
                  failure-only notifications. Falls back to NOTIFY_STEP_MUTE.
    """
    messages = messages or {}
    active_mute = set(m.lower() for m in (mute or DEFAULT_MUTE))
    t0 = time.time()

    ctx: Dict[str, Any] = {
        "title": title,
        "log_path": log_path,
        "started_at": t0,
        "duration_s": None,
        "exit_code": None,
        "exc": None,
    }

    if "start" not in active_mute:
        notify("start", title, body=_render(messages.get("start"), ctx, f"Starting: {title}"))

    try:
        yield

    except SystemExit as exc:
        code = int(getattr(exc, "code", 0) or 0)
        if code != 0:
            ctx.update(exit_code=code, exc=None, duration_s=int(time.time() - t0))
            if "fail" not in active_mute:
                notify(
                    "fail",
                    f"{title} failed",
                    body=_render(messages.get("fail"), ctx, f"Exit code: {code}"),
                    log_tail=read_tail(log_path),
                )
        raise

    except Exception as exc:
        ctx.update(exit_code=None, exc=exc, duration_s=int(time.time() - t0))
        if "fail" not in active_mute:
            notify(
                "fail",
                f"{title} crashed",
                body=_render(messages.get("fail"), ctx, f"{type(exc).__name__}: {exc}"),
                log_tail=read_tail(log_path),
            )
        raise

    else:
        ctx.update(duration_s=int(time.time() - t0), exit_code=0, exc=None)
        if "ok" not in active_mute:
            notify(
                "ok",
                f"{title} succeeded",
                body=_render(messages.get("ok"), ctx, f"Duration: {ctx['duration_s']}s"),
                log_tail=read_tail(log_path),
            )


# ---------------------------------------------------------------------------
# Channel implementations
# ---------------------------------------------------------------------------


def _notify_email(status: str, title: str, body: str, log_tail: Optional[str]) -> None:
    """Send an email notification via SMTP if credentials are configured."""
    if not (_EMAIL_TO and _SMTP_HOST and _SMTP_USER and _SMTP_PASSWORD):
        return

    subject = f"[{status.upper()}] {title}"

    parts = [body] if body else []
    if log_tail:
        parts.append("\n--- Log tail ---\n" + log_tail)
    text = "\n\n".join(parts) or title

    msg = MIMEMultipart()
    msg["From"] = _EMAIL_FROM
    msg["To"] = _EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(text, "plain"))

    try:
        if _SMTP_TLS:
            with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(_SMTP_USER, _SMTP_PASSWORD)
                smtp.sendmail(_EMAIL_FROM, _EMAIL_TO, msg.as_string())
        else:
            with smtplib.SMTP_SSL(_SMTP_HOST, _SMTP_PORT) as smtp:
                smtp.login(_SMTP_USER, _SMTP_PASSWORD)
                smtp.sendmail(_EMAIL_FROM, _EMAIL_TO, msg.as_string())
    except Exception:
        pass


def _notify_slack(status: str, title: str, body: str, log_tail: Optional[str]) -> None:
    """Post a Slack message via incoming webhook if configured."""
    if not _SLACK:
        return

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": title}},
        {"type": "section", "text": {"type": "mrkdwn", "text": body or ""}},
    ]
    if log_tail:
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": "```\n" + log_tail + "\n```"}}
        )

    payload = {"attachments": [{"color": _STATUS_COLORS.get(status, "#888888"), "blocks": blocks}]}
    try:
        requests.post(_SLACK, json=payload, timeout=8)
    except Exception:
        pass


def _notify_healthchecks(status: str, body: str) -> None:
    """Ping the appropriate Healthchecks.io URL for the given status."""
    url = None
    params = None

    if status == "start" and _HC_START:
        url = _HC_START
    elif status == "ok" and _HC_SUCCESS:
        url = _HC_SUCCESS
    elif status == "fail" and _HC_FAIL:
        url = _HC_FAIL
        params = {"msg": (body or "")[:400]}

    if not url:
        return

    try:
        full_url = f"{url}?{urlencode(params)}" if params else url
        requests.get(full_url, timeout=8)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render(spec: MessageSpec, ctx: Dict[str, Any], default: str) -> str:
    """Resolve a MessageSpec to a plain string."""
    if spec is None:
        return default
    return spec(ctx) if callable(spec) else str(spec)
