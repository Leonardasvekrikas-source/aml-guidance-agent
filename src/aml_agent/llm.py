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


def _resolved_key(client: anthropic.Anthropic) -> str:
    """The credential the SDK actually resolved, if any.

    Constructing the client is NOT proof of credentials: with no key anywhere,
    anthropic.Anthropic() constructs happily with api_key set to the empty
    string and only fails later, at request time, as a 401 that says nothing
    about which setup step was missed.
    """
    for attribute in ("api_key", "auth_token"):
        value = getattr(client, attribute, None)
        if value:
            return str(value)
    return ""


def make_client() -> anthropic.Anthropic:
    if settings.anthropic_api_key:
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)

    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 - re-raised with a usable message
        raise MissingCredentials(_SETUP_HELP) from exc

    if not _resolved_key(client):
        raise MissingCredentials(_SETUP_HELP)
    return client


def have_credentials() -> bool:
    """Whether an LLM-backed step can actually run.

    Used by the evaluation harness and /health so that a missing key produces a
    clear "this cannot run" rather than forty identical 401s.
    """
    try:
        make_client()
        return True
    except MissingCredentials:
        return False
