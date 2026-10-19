"""Tests for pipelines/notify_failure.py with a fake SMTP server (nothing is sent)."""

import pytest

import pipelines.notify_failure as nf

ENV = {"ALERT_SENDER_EMAIL": "bot@example.com", "ALERT_RECEIVER_EMAIL": "me@example.com, ops@example.com",
       "GMAIL_APP_PASSWORD": "abcd efgh ijkl mnop"}


class FakeSMTP:
    instances = []

    def __init__(self, host, port, context=None, timeout=None):
        self.host, self.port, self.context, self.timeout = host, port, context, timeout
        self.login_args = self.sent = None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        self.login_args = (user, password)

    def send_message(self, msg):
        self.sent = msg


@pytest.fixture()
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"):
        monkeypatch.delenv(k, raising=False)
    FakeSMTP.instances.clear()
    monkeypatch.setattr(nf.smtplib, "SMTP_SSL", FakeSMTP)


def test_sends_over_ssl_with_app_password_spaces_removed(env):
    nf.send_failure_alert("1 failed", failed_step="pytest")
    smtp = FakeSMTP.instances[0]
    assert (smtp.host, smtp.port) == ("smtp.gmail.com", 465)
    assert smtp.context is not None
    assert smtp.login_args == ("bot@example.com", "abcdefghijklmnop")
    assert smtp.sent["To"] == "me@example.com, ops@example.com"
    assert "pytest" in smtp.sent["Subject"]
    assert "1 failed" in smtp.sent.get_content()


def test_missing_secrets_raise_config_error(env, monkeypatch):
    monkeypatch.delenv("GMAIL_APP_PASSWORD")
    with pytest.raises(nf.AlertConfigError, match="GMAIL_APP_PASSWORD"):
        nf.send_failure_alert("x")
    assert FakeSMTP.instances == []


def test_cli_exits_2_without_secrets(env, monkeypatch, capsys):
    monkeypatch.delenv("ALERT_SENDER_EMAIL")
    assert nf.main(["--message", "boom"]) == 2
    assert "::warning::" in capsys.readouterr().out


def test_cli_exits_1_when_smtp_fails(env, monkeypatch):
    class Refusing(FakeSMTP):
        def login(self, user, password):
            raise nf.smtplib.SMTPAuthenticationError(535, b"bad credentials")
    monkeypatch.setattr(nf.smtplib, "SMTP_SSL", Refusing)
    assert nf.main(["--message", "boom"]) == 1


def test_github_run_url_is_included(env, monkeypatch):
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "me/retail")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    msg = nf.build_message("log", "a@b.c", ["d@e.f"])
    assert "https://github.com/me/retail/actions/runs/42" in msg.get_content()


def test_long_logs_are_tailed():
    text = "\n".join(f"line {i}" for i in range(500))
    out = nf.tail(text, 10)
    assert out.splitlines()[0] == "[... 490 earlier lines omitted ...]"
    assert out.splitlines()[-1] == "line 499"


def test_dry_run_prints_without_sending(env, tmp_path, capsys):
    log = tmp_path / "pytest.txt"
    log.write_text("FAILED tests/test_portfolio_health.py::test_database_exists")
    assert nf.main(["--log-file", str(log), "--failed-step", "pytest", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "test_database_exists" in out and FakeSMTP.instances == []
    assert "abcd" not in out  # password never printed


def test_missing_log_file_is_reported_in_body(env, tmp_path, capsys):
    assert nf.main(["--log-file", str(tmp_path / "nope.txt"), "--dry-run"]) == 0
    assert "not found" in capsys.readouterr().out
