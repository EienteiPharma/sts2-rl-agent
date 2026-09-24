"""TypeSafe HTTP transient DNS/network retry wiring."""

from __future__ import annotations

import socket
import urllib.error
import urllib.request

import pytest

from sts2_env.eval.jev_client import (
    LiveJevClient,
    TYPESAFE_NETWORK_RETRY_MAX,
    _is_transient_network_url_error,
)
from sts2_env.eval.jev_types import JevError


def _dns_url_error() -> urllib.error.URLError:
    return urllib.error.URLError(
        socket.gaierror(-3, "Temporary failure in name resolution")
    )


def test_is_transient_network_url_error_dns():
    assert _is_transient_network_url_error(_dns_url_error())
    assert not _is_transient_network_url_error(
        urllib.error.URLError("connection refused")
    )


def test_transient_url_error_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class FakeResp:
        def read(self):
            return b'{"answers": {"pick": {"choice": "a"}}}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _dns_url_error()
        return FakeResp()

    client = LiveJevClient(api_key="k")
    monkeypatch.setattr(client, "_try_sdk", lambda *a, **k: None)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_env.eval.jev_client.time.sleep", lambda s: None)

    payload = client._call({}, {"pick": {"type": "choice"}})
    assert calls["n"] == 2
    assert payload["answers"]["pick"]["choice"] == "a"


def test_transient_url_error_exhaust_raises_jev_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise _dns_url_error()

    client = LiveJevClient(api_key="k")
    monkeypatch.setattr(client, "_try_sdk", lambda *a, **k: None)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_env.eval.jev_client.time.sleep", lambda s: None)

    with pytest.raises(JevError, match="typesafe network error"):
        client._call({}, {"pick": {"type": "choice"}})


def test_transient_url_error_retry_count(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _dns_url_error()

    client = LiveJevClient(api_key="k")
    monkeypatch.setattr(client, "_try_sdk", lambda *a, **k: None)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_env.eval.jev_client.time.sleep", lambda s: None)

    with pytest.raises(JevError):
        client._call({}, {"pick": {"type": "choice"}})

    assert calls["n"] == TYPESAFE_NETWORK_RETRY_MAX
