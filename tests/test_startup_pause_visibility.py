"""Startup tests for persisted paper pause-state visibility.

``main()`` must report the pause control state without ever resuming
trading or rewriting the control file. Three cases are pinned here: no
control file (not manually paused), an explicit pause (a warning that
names the reason) and an unreadable control file (a warning that new
entries are blocked). The file must be byte-identical after startup in
every case.
"""

from __future__ import annotations

import json

import main as main_module

PAUSE_FILE = ("logs", "paper_pause.json")


def _run_main(monkeypatch, tmp_path) -> None:
    """Run ``main()`` with the bot start stubbed out."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module.settings, "TELEGRAM_BOT_TOKEN", "123:abc")
    # The execution-mode safety check is a classmethod, so pin the class
    # attributes: other tests change them and must not decide startup here.
    settings_type = type(main_module.settings)
    monkeypatch.setattr(settings_type, "EXECUTION_MODE", "PAPER")
    monkeypatch.setattr(settings_type, "PAPER_TRADING", True)
    monkeypatch.setattr(main_module, "validate_settings", lambda: [])

    started = []
    monkeypatch.setattr(
        "telegram_bot.bot.run_bot", lambda: started.append(True)
    )

    main_module.main()

    assert started == [True], "startup must reach the bot start"


def _pause_path(tmp_path):
    path = tmp_path.joinpath(*PAUSE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_missing_pause_file_reports_not_manually_paused(
    monkeypatch, tmp_path, capsys
) -> None:
    _run_main(monkeypatch, tmp_path)

    out = capsys.readouterr().out
    assert "not manually paused" in out
    assert "PAUSED" not in out
    assert "BLOCKED" not in out
    # A missing control file must stay missing: startup never creates it.
    assert not tmp_path.joinpath(*PAUSE_FILE).exists()


def test_paused_state_prints_a_warning_and_keeps_the_file(
    monkeypatch, tmp_path, capsys
) -> None:
    path = _pause_path(tmp_path)
    original = json.dumps(
        {"paused": True, "reason": "maintenance window", "updated_at": "now"},
        indent=2,
    )
    path.write_text(original, encoding="utf-8")

    _run_main(monkeypatch, tmp_path)

    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert "manually PAUSED" in out
    assert "maintenance window" in out
    assert "blocked" in out
    # Startup must never resume or rewrite the control file.
    assert path.read_text(encoding="utf-8") == original


def test_corrupt_pause_state_warns_that_entries_are_blocked(
    monkeypatch, tmp_path, capsys
) -> None:
    path = _pause_path(tmp_path)
    original = "{ this is not json"
    path.write_text(original, encoding="utf-8")

    _run_main(monkeypatch, tmp_path)

    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert "BLOCKED" in out
    assert "cannot be trusted" in out
    # Unreadable state is never "repaired" or reset by startup.
    assert path.read_text(encoding="utf-8") == original
