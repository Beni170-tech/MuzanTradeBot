"""
Tests for the structured, rotating trade journal (Milestone M-C).

The journal must record structured records, rotate by size, and never
raise into the /analyze pipeline: a failed write is reported through the
return value and journal_status() instead of breaking the reply.
"""

from __future__ import annotations

import json
from pathlib import Path

from journal.journal import (
    append_entry,
    journal_status,
    log_trade,
    rotated_path,
)


def _records(path: Path) -> list[dict]:
    """Parse a JSONL journal file into records."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_log_trade_writes_one_structured_record(tmp_path):
    journal = tmp_path / "trade_journal.jsonl"

    assert log_trade("BTC", "BUY", "APPROVED", path=journal) is True

    records = _records(journal)
    assert len(records) == 1
    record = records[0]
    assert record["symbol"] == "BTC"
    assert record["decision"] == "BUY"
    assert record["risk_verdict"] == "APPROVED"
    assert len(record["timestamp"]) == len("2025-01-01 00:00:00")


def test_appends_records_in_order(tmp_path):
    journal = tmp_path / "trade_journal.jsonl"

    log_trade("BTC", "BUY", "APPROVED", path=journal)
    log_trade("ETH", "SELL", "REJECTED", path=journal)

    assert [record["symbol"] for record in _records(journal)] == ["BTC", "ETH"]


def test_legacy_plain_text_journal_is_not_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    journal = tmp_path / "logs" / "trade_journal.jsonl"

    log_trade("BTC", "BUY", "APPROVED", path=journal)

    assert journal.exists()
    assert not (tmp_path / "logs" / "trade_journal.txt").exists()


def test_rotation_keeps_bounded_history(tmp_path):
    journal = tmp_path / "trade_journal.jsonl"

    for index in range(10):
        append_entry(
            {"n": index, "padding": "x" * 40},
            path=journal,
            max_bytes=120,
            backup_count=2,
        )

    assert rotated_path(journal, 1).exists()
    assert rotated_path(journal, 2).exists()
    assert not rotated_path(journal, 3).exists()

    # Every surviving line is still a complete JSON record, and the
    # newest entries are the ones kept.
    surviving: list[dict] = []
    for candidate in (
        rotated_path(journal, 2),
        rotated_path(journal, 1),
        journal,
    ):
        if candidate.exists():
            surviving.extend(_records(candidate))
    assert surviving
    assert surviving[-1]["n"] == 9
    assert [entry["n"] for entry in surviving] == sorted(
        entry["n"] for entry in surviving
    )


def test_rotation_with_zero_backups_discards_old_file(tmp_path):
    journal = tmp_path / "trade_journal.jsonl"

    append_entry({"n": 1, "padding": "x" * 40}, path=journal, max_bytes=60)
    append_entry(
        {"n": 2, "padding": "x" * 40},
        path=journal,
        max_bytes=60,
        backup_count=0,
    )

    assert _records(journal) == [{"n": 2, "padding": "x" * 40}]
    assert not rotated_path(journal, 1).exists()


def test_oversized_record_is_still_written_once(tmp_path):
    journal = tmp_path / "trade_journal.jsonl"
    entry = {"n": 1, "padding": "x" * 200}

    assert append_entry(entry, path=journal, max_bytes=10) is True
    assert _records(journal) == [entry]


def test_write_failure_is_non_blocking_and_reported(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    result = log_trade("BTC", "BUY", "APPROVED", path=blocker / "journal.jsonl")

    assert result is False
    status = journal_status(blocker / "journal.jsonl")
    assert status.exists is False
    assert status.last_error is not None
    assert "blocker" in status.last_error


def test_successful_write_clears_previous_error(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    assert log_trade("BTC", "BUY", "APPROVED", path=blocker / "j.jsonl") is False

    journal = tmp_path / "trade_journal.jsonl"
    assert log_trade("BTC", "BUY", "APPROVED", path=journal) is True

    assert journal_status(journal).last_error is None


def test_journal_status_reports_size_rotation_and_last_write(tmp_path):
    journal = tmp_path / "trade_journal.jsonl"

    empty = journal_status(journal)
    assert empty.exists is False
    assert empty.size_bytes == 0
    assert empty.last_write_at is None
    assert empty.rotated_files == 0

    append_entry({"n": 1, "padding": "x" * 40}, path=journal, max_bytes=60)
    append_entry({"n": 2, "padding": "x" * 40}, path=journal, max_bytes=60)

    status = journal_status(journal)
    assert status.exists is True
    assert status.size_bytes == journal.stat().st_size
    assert status.last_write_at is not None
    assert status.rotated_files >= 1
