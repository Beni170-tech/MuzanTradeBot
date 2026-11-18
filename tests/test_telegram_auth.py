"""Tests for telegram_bot.auth (Milestone M-A access control).

The gate must protect every update centrally and deny by default. These
tests pin the pure allowlist logic, the update decision, the PTB gate
(denial notice, propagation stop, fail-safe reply) and the registration
group that makes the gate run before command handlers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from telegram.ext import ApplicationHandlerStop

from config.settings import settings
from telegram_bot import auth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, fail: bool = False):
        self.messages: list[str] = []
        self.fail = fail

    async def __call__(self, text):
        if self.fail:
            raise RuntimeError("telegram is down")
        self.messages.append(text)


def _update(chat_id=111, user_id=222, message=None):
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
        effective_message=message,
    )


def _message(fail=False):
    recorder = _Recorder(fail=fail)
    return SimpleNamespace(reply_text=recorder), recorder


# ---------------------------------------------------------------------------
# parse_allowlist
# ---------------------------------------------------------------------------


class TestParseAllowlist:
    def test_missing_value_is_empty(self):
        assert auth.parse_allowlist(None) == frozenset()
        assert auth.parse_allowlist("") == frozenset()

    def test_comma_separated_ids(self):
        assert auth.parse_allowlist("111,222") == frozenset({111, 222})

    def test_whitespace_and_newlines(self):
        assert auth.parse_allowlist(" 111 \n 222\t333 ") == frozenset(
            {111, 222, 333}
        )

    def test_negative_group_chat_ids_are_valid(self):
        assert auth.parse_allowlist("-1001234567890") == frozenset(
            {-1001234567890}
        )

    def test_iterable_of_ints(self):
        assert auth.parse_allowlist([111, "222"]) == frozenset({111, 222})

    def test_invalid_entries_are_dropped(self):
        assert auth.parse_allowlist("111,abc,222") == frozenset({111, 222})

    def test_only_invalid_entries_lock_the_bot(self):
        assert auth.parse_allowlist("abc,,nope") == frozenset()

    def test_zero_is_not_an_id(self):
        assert auth.parse_allowlist("0,111") == frozenset({111})

    def test_booleans_are_not_ids(self):
        assert auth.parse_allowlist([True, False, 111]) == frozenset({111})


# ---------------------------------------------------------------------------
# is_authorized
# ---------------------------------------------------------------------------


class TestIsAuthorized:
    def test_empty_allowlist_denies_even_a_matching_id(self):
        assert auth.is_authorized(_update(111, 222), frozenset()) is False

    def test_user_id_match_authorizes(self):
        assert auth.is_authorized(
            _update(chat_id=999, user_id=222), frozenset({222})
        )

    def test_chat_id_match_authorizes(self):
        assert auth.is_authorized(
            _update(chat_id=111, user_id=999), frozenset({111})
        )

    def test_unrelated_ids_are_denied(self):
        assert not auth.is_authorized(
            _update(chat_id=999, user_id=888), frozenset({111, 222})
        )

    def test_update_without_ids_is_denied(self):
        assert not auth.is_authorized(SimpleNamespace(), frozenset({111}))

    def test_boolean_ids_are_not_treated_as_matches(self):
        update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=True),
            effective_user=SimpleNamespace(id=True),
        )
        assert not auth.is_authorized(update, frozenset({1}))


# ---------------------------------------------------------------------------
# authorization_gate
# ---------------------------------------------------------------------------


class TestAuthorizationGate:
    @pytest.mark.asyncio
    async def test_authorized_update_passes_without_reply(
        self, monkeypatch
    ):
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", "222")
        message, recorder = _message()
        await auth.authorization_gate(
            _update(chat_id=999, user_id=222, message=message), None
        )
        assert recorder.messages == []

    @pytest.mark.asyncio
    async def test_unauthorized_update_is_denied_and_stopped(
        self, monkeypatch
    ):
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", "222")
        message, recorder = _message()
        with pytest.raises(ApplicationHandlerStop):
            await auth.authorization_gate(
                _update(chat_id=999, user_id=888, message=message), None
            )
        assert recorder.messages == [auth.DENIED_MESSAGE]

    @pytest.mark.asyncio
    async def test_missing_allowlist_denies_by_default(self, monkeypatch):
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", "")
        message, recorder = _message()
        with pytest.raises(ApplicationHandlerStop):
            await auth.authorization_gate(
                _update(chat_id=111, user_id=222, message=message), None
            )
        assert recorder.messages == [auth.DENIED_MESSAGE]

    @pytest.mark.asyncio
    async def test_allowlist_is_read_at_call_time(self, monkeypatch):
        message, recorder = _message()
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", "")
        with pytest.raises(ApplicationHandlerStop):
            await auth.authorization_gate(
                _update(111, 222, message), None
            )

        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", "111")
        await auth.authorization_gate(_update(111, 222, message), None)
        assert len(recorder.messages) == 1

    @pytest.mark.asyncio
    async def test_reply_failure_still_stops_the_update(
        self, monkeypatch
    ):
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", "222")
        message, _ = _message(fail=True)
        with pytest.raises(ApplicationHandlerStop):
            await auth.authorization_gate(
                _update(chat_id=999, user_id=888, message=message), None
            )

    @pytest.mark.asyncio
    async def test_update_without_message_is_stopped_silently(
        self, monkeypatch
    ):
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", "222")
        with pytest.raises(ApplicationHandlerStop):
            await auth.authorization_gate(_update(999, 888, None), None)


# ---------------------------------------------------------------------------
# Registration and status
# ---------------------------------------------------------------------------


class _FakeApplication:
    def __init__(self):
        self.calls = []

    def add_handler(self, handler, group=0):
        self.calls.append((handler, group))


class TestRegistration:
    def test_gate_is_registered_in_the_first_group(self):
        app = _FakeApplication()
        auth.install_authorization_gate(app)
        assert len(app.calls) == 1
        handler, group = app.calls[0]
        assert group == -1
        assert handler.callback is auth.authorization_gate

    def test_status_never_prints_ids(self):
        assert "5" not in auth.authorization_status("555")
        assert "authorized ID(s)" in auth.authorization_status("555,666")
        assert "locked" in auth.authorization_status("")

    def test_status_defaults_to_live_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", "777")
        assert auth.authorization_status() == "1 authorized ID(s)"
