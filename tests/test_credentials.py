"""Credential detection.

Regression test for a real bug: an empty API key was treated as valid because
the SDK client constructs without raising, so /health reported credentials
present and /ask returned a 500 instead of setup instructions.
"""

from __future__ import annotations

import pytest

from aml_agent import llm


def _settings_with_key(key: str):
    """Settings is a frozen dataclass, so build a replacement rather than
    mutating it."""
    from dataclasses import replace

    return replace(llm.settings, anthropic_api_key=key)


class FakeClient:
    def __init__(self, api_key="", auth_token=None):
        self.api_key = api_key
        self.auth_token = auth_token


class TestResolvedKey:
    def test_empty_key_is_not_a_credential(self):
        assert llm._resolved_key(FakeClient(api_key="")) == ""

    def test_none_key_is_not_a_credential(self):
        assert llm._resolved_key(FakeClient(api_key=None)) == ""

    def test_real_key_is_found(self):
        assert llm._resolved_key(FakeClient(api_key="sk-ant-xyz")) == "sk-ant-xyz"

    def test_auth_token_counts_when_api_key_is_absent(self):
        assert llm._resolved_key(FakeClient(api_key="", auth_token="tok")) == "tok"


class TestMakeClient:
    def test_empty_resolution_raises_with_setup_steps(self, monkeypatch):
        monkeypatch.setattr(llm, "settings", _settings_with_key(""))
        monkeypatch.setattr(llm.anthropic, "Anthropic", lambda **kw: FakeClient(api_key=""))

        with pytest.raises(llm.MissingCredentials) as excinfo:
            llm.make_client()

        # The message must name the fix, not just the failure.
        assert "ANTHROPIC_API_KEY" in str(excinfo.value)
        assert ".env" in str(excinfo.value)

    def test_have_credentials_is_false_when_unresolved(self, monkeypatch):
        monkeypatch.setattr(llm, "settings", _settings_with_key(""))
        monkeypatch.setattr(llm.anthropic, "Anthropic", lambda **kw: FakeClient(api_key=""))
        assert llm.have_credentials() is False

    def test_have_credentials_is_true_with_a_key(self, monkeypatch):
        monkeypatch.setattr(llm, "settings", _settings_with_key("sk-ant-real"))
        monkeypatch.setattr(
            llm.anthropic, "Anthropic", lambda **kw: FakeClient(api_key="sk-ant-real")
        )
        assert llm.have_credentials() is True
