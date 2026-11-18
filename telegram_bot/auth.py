"""Telegram access control (Milestone M-A).

A single, shared authorization gate for every Telegram update the bot
receives. Access is denied by default: when the allowlist is missing,
empty or contains no usable numeric id, nobody is authorized.

Configuration:
    ``TELEGRAM_ALLOWED_CHAT_IDS`` holds comma-separated Telegram user ids
    and/or chat ids. Matching either the update's effective user id or
    its effective chat id authorizes the update, so a private chat can be
    configured by user id and a group by chat id.

Design notes:
    - The allowlist is re-read from settings at call time, so a patched
      or reloaded configuration applies to the next update without a
      restart and tests can patch one value.
    - Invalid entries are ignored rather than treated as wildcards; an
      allowlist that parses to nothing locks the bot instead of opening
      it.
    - Authorization happens in one place, before any command handler,
      so new commands and future callback handlers are covered without
      having to remember a per-handler check.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from telegram import Update
from telegram.ext import ApplicationHandlerStop, TypeHandler

from config.settings import settings

logger = logging.getLogger(__name__)

# Sent to any update that is not authorized. It deliberately contains no
# account, limit or configuration information.
DENIED_MESSAGE = "⛔ Unauthorized. This bot is private."


# ---------------------------------------------------------------------------
# Pure allowlist logic
# ---------------------------------------------------------------------------


def parse_allowlist(raw: str | Iterable[Any] | None) -> frozenset[int]:
    """Parse an allowlist into a frozen set of Telegram ids.

    Args:
        raw: A comma/whitespace separated string, an iterable of values,
            or None.

    Returns:
        The valid, non-zero integer ids. Invalid entries are dropped, so
        a malformed value yields an empty set (locked), never a
        wildcard.
    """
    if raw is None:
        return frozenset()

    if isinstance(raw, str):
        tokens: Iterable[Any] = raw.replace(",", " ").split()
    else:
        try:
            tokens = list(raw)
        except TypeError:
            return frozenset()

    ids: set[int] = set()
    for token in tokens:
        if isinstance(token, bool):
            # bool is an int subclass; True/False are not Telegram ids.
            continue
        if isinstance(token, int):
            value = token
        else:
            try:
                value = int(str(token).strip())
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid TELEGRAM_ALLOWED_CHAT_IDS entry %r.",
                    token,
                )
                continue
        if value == 0:
            logger.warning(
                "Ignoring invalid TELEGRAM_ALLOWED_CHAT_IDS entry 0."
            )
            continue
        ids.add(value)

    return frozenset(ids)


def _update_ids(update: Any) -> tuple[int | None, int | None]:
    """Return ``(chat_id, user_id)`` from an update, tolerating fakes."""
    chat = getattr(update, "effective_chat", None)
    user = getattr(update, "effective_user", None)
    chat_id = getattr(chat, "id", None)
    user_id = getattr(user, "id", None)
    return (
        chat_id if isinstance(chat_id, int) and not isinstance(chat_id, bool)
        else None,
        user_id if isinstance(user_id, int) and not isinstance(user_id, bool)
        else None,
    )


def is_authorized(update: Any, allowlist: frozenset[int]) -> bool:
    """Return True when ``update`` comes from an authorized id.

    An empty allowlist always denies. The update is authorized when
    either its effective chat id or its effective user id is listed.
    """
    if not allowlist:
        return False
    chat_id, user_id = _update_ids(update)
    return any(
        candidate is not None and candidate in allowlist
        for candidate in (chat_id, user_id)
    )


def current_allowlist() -> frozenset[int]:
    """Return the allowlist described by the live settings."""
    return parse_allowlist(settings.TELEGRAM_ALLOWED_CHAT_IDS)


def authorization_status(raw: str | Iterable[Any] | None = None) -> str:
    """Return a secret-free authorization status line for /settings.

    Args:
        raw: Optional allowlist value. Defaults to the live settings.

    Returns:
        A short status string; ids themselves are never printed.
    """
    ids = parse_allowlist(
        raw if raw is not None else settings.TELEGRAM_ALLOWED_CHAT_IDS
    )
    if not ids:
        return "locked (no authorized IDs configured)"
    return f"{len(ids)} authorized ID(s)"


# ---------------------------------------------------------------------------
# Update gate
# ---------------------------------------------------------------------------


async def authorization_gate(update: Any, context: Any) -> None:
    """Deny unauthorized updates before any command handler runs.

    Authorized updates return immediately and continue to the normal
    handlers. Unauthorized updates receive the denial notice (best
    effort) and are stopped from propagating, so no account information
    is read and no command executes.

    Raises:
        ApplicationHandlerStop: Always, for unauthorized updates.
    """
    allowlist = current_allowlist()
    if is_authorized(update, allowlist):
        return

    chat_id, user_id = _update_ids(update)
    logger.warning(
        "Denied unauthorized Telegram update (chat_id=%r, user_id=%r); "
        "allowlist has %d id(s).",
        chat_id,
        user_id,
        len(allowlist),
    )

    message = getattr(update, "effective_message", None)
    reply = getattr(message, "reply_text", None)
    if callable(reply):
        try:
            await reply(DENIED_MESSAGE)
        except Exception:
            logger.warning(
                "Could not send the authorization denial notice.",
                exc_info=True,
            )

    raise ApplicationHandlerStop


def install_authorization_gate(application: Any) -> None:
    """Install the gate as the first handler on ``application``.

    PTB runs lower-numbered handler groups first. Group -1 ensures the
    gate sees every update before any command or callback handler, and
    ``ApplicationHandlerStop`` prevents the rest from running.
    """
    application.add_handler(
        TypeHandler(Update, authorization_gate), group=-1
    )
