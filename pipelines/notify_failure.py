"""Email a pipeline-failure alert through Gmail SMTP over SSL (port 465).

Credentials come from environment variables (GitHub Actions secrets in CI):
    ALERT_SENDER_EMAIL     Gmail address the alert is sent from
    ALERT_RECEIVER_EMAIL   where the alert goes (comma-separate for several)
    GMAIL_APP_PASSWORD     a 16-character Gmail App Password for the sender, NOT the account
                           password (Google Account -> Security -> 2-Step Verification -> App passwords)

In GitHub Actions the alert includes a link to the failed run, built from the standard
GITHUB_SERVER_URL / GITHUB_REPOSITORY / GITHUB_RUN_ID variables.

Usage:
    python pipelines/notify_failure.py --log-file health_logs/pytest.txt
    python pipelines/notify_failure.py --log-file health_logs/pytest.txt --dry-run   # print, don't send
"""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
LOG_TAIL_LINES = 120          # enough for a pytest short summary without flooding the inbox
REQUIRED_ENV = ("ALERT_SENDER_EMAIL", "ALERT_RECEIVER_EMAIL", "GMAIL_APP_PASSWORD")


class AlertConfigError(RuntimeError):
    """Raised when the alert can't be sent because configuration is missing."""


def run_url() -> str | None:
    env = os.environ
    if all(env.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")):
        return f"{env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}/actions/runs/{env['GITHUB_RUN_ID']}"
    return None


def tail(text: str, n: int = LOG_TAIL_LINES) -> str:
    lines = text.rstrip().splitlines()
    if len(lines) <= n:
        return "\n".join(lines)
    return f"[... {len(lines) - n} earlier lines omitted ...]\n" + "\n".join(lines[-n:])


def build_message(error_log: str, sender: str, receivers: list[str],
                  failed_step: str | None = None) -> EmailMessage:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    url = run_url()
    msg = EmailMessage()
    msg["Subject"] = "🚨 Portfolio Health Alert: E-Commerce Retention Pipeline failed" + (
        f" at '{failed_step}'" if failed_step else "")
    msg["From"] = sender
    msg["To"] = ", ".join(receivers)
    body = [
        "The automated health check for the E-Commerce Customer Segmentation & Retention Engine failed.",
        "",
        f"When:        {when}",
        f"Failed step: {failed_step or 'see log'}",
        f"Run:         {url or 'local run (no GitHub Actions URL)'}",
        "",
        "---- log tail ----",
        tail(error_log) if error_log.strip() else "(no log captured)",
        "------------------",
        "",
        "Full logs and the JUnit report are attached to the workflow run as artifacts.",
    ]
    msg.set_content("\n".join(body))
    return msg


def read_config() -> tuple[str, list[str], str]:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise AlertConfigError(f"Missing environment variables: {', '.join(missing)}")
    receivers = [r.strip() for r in os.environ["ALERT_RECEIVER_EMAIL"].split(",") if r.strip()]
    # App passwords are shown with spaces ("abcd efgh ijkl mnop"); SMTP wants them without.
    password = os.environ["GMAIL_APP_PASSWORD"].replace(" ", "")
    return os.environ["ALERT_SENDER_EMAIL"].strip(), receivers, password


def send_failure_alert(error_log: str, failed_step: str | None = None, timeout: int = 30) -> None:
    """Send the alert. Raises AlertConfigError or smtplib/OSError errors on failure."""
    sender, receivers, password = read_config()
    msg = build_message(error_log, sender, receivers, failed_step)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=timeout) as server:
        server.login(sender, password)
        server.send_message(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-file", type=Path, help="log whose tail goes in the email")
    parser.add_argument("--message", default="", help="text to send if there's no log file")
    parser.add_argument("--failed-step", help="name of the step that failed")
    parser.add_argument("--dry-run", action="store_true", help="print the email instead of sending")
    args = parser.parse_args(argv)

    error_log = args.message
    if args.log_file:
        error_log = (args.log_file.read_text(errors="replace") if args.log_file.exists()
                     else f"(log file {args.log_file} not found: the step probably failed before writing it)")

    if args.dry_run:
        sender = os.environ.get("ALERT_SENDER_EMAIL", "sender@example.com")
        receivers = [os.environ.get("ALERT_RECEIVER_EMAIL", "receiver@example.com")]
        msg = build_message(error_log, sender, receivers, args.failed_step)
        # Print the decoded body, not str(msg): the MIME form soft-wraps long lines.
        print(f"Subject: {msg['Subject']}\nFrom: {msg['From']}\nTo: {msg['To']}\n\n{msg.get_content()}")
        return 0

    try:
        send_failure_alert(error_log, args.failed_step)
    except AlertConfigError as exc:
        print(f"::warning::Alert not sent: {exc}. Add them as repository secrets to enable email alerts.")
        return 2
    except (smtplib.SMTPException, OSError) as exc:
        # Never echo the password; the exception text from smtplib doesn't contain it.
        print(f"::error::Failed to send alert email: {exc}", file=sys.stderr)
        return 1
    print("Alert email sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
