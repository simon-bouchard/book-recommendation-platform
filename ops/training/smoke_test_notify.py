# ops/training/smoke_test_notify.py
"""
Manual smoke test for the notification pipeline.

Sends one notification for each status (start, ok, fail) so you can verify
that emails arrive correctly. Not part of the pytest suite — run by hand only.

Usage:
    python ops/training/smoke_test_notify.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from ops.training.notify import notify, _EMAIL_TO, _SMTP_HOST, _SMTP_USER


def main() -> None:
    if not (_EMAIL_TO and _SMTP_HOST and _SMTP_USER):
        print(
            "Email not configured. Set NOTIFY_EMAIL_TO, NOTIFY_SMTP_HOST, "
            "NOTIFY_SMTP_USER, and NOTIFY_SMTP_PASSWORD in your .env file."
        )
        sys.exit(1)

    print(f"Sending test notifications to: {_EMAIL_TO}")

    print("  Sending start...")
    notify("start", "Smoke test", body="Pipeline is starting.")

    print("  Sending ok...")
    notify("ok", "Smoke test", body="Pipeline completed successfully. Duration: 42s")

    print("  Sending fail...")
    notify(
        "fail",
        "Smoke test",
        body="Something went wrong during training.",
        log_tail="Traceback (most recent call last):\n  ...\nValueError: recall below floor",
    )

    print("Done. Check your inbox.")


if __name__ == "__main__":
    main()
