"""Anthropic client construction, in one place.

The SDK resolves credentials from several sources — ANTHROPIC_API_KEY, an
ANTHROPIC_AUTH_TOKEN, or a profile written by `ant auth login`. Hardcoding the
env var as the only source would refuse to run on a machine that is in fact
authenticated, so an explicit key is used when present and the SDK's own
resolution is trusted otherwise.

The failure message matters. "anthropic.AuthenticationError" tells a stranger
who cloned this repository nothing about which of the two setup steps they
missed.
"""

from __future__ import annotations

import anthropic

from .config import settings

_SETUP_HELP = (
    "No Anthropic credentials found.\n"
    "  1. cp .env.example .env\n"
    "  2. add ANTHROPIC_API_KEY=sk-ant-... to .env\n"
    "  3. re-run\n"
    "Retrieval and `make eval-retrieval` do not need a key. The agent loop, "
    "citation support checking, the LLM judge and `make ask` do."
)


class MissingCredentials(RuntimeError):
    pass


def make_client() -> anthropic.Anthropic:
    if settings.anthropic_api_key:
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # No explicit key. The SDK may still find a profile or auth token; if it
    # cannot, it raises at construction time and we translate the message.
    try:
        return anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 - re-raised with a usable message
        raise MissingCredentials(_SETUP_HELP) from exc


def have_credentials() -> bool:
    """Whether an LLM-backed step can run at all.

    Used by the evaluation harness so that a missing key produces a clear
    "this metric was not computed" rather than forty identical stack traces.
    """
    if settings.anthropic_api_key:
        return True
    try:
        anthropic.Anthropic()
        return True
    except Exception:  # noqa: BLE001
        return False
