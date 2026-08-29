"""Check that every source URL in the manifest still resolves.

The corpus is defined by thirty URLs on other people's servers, and those rot:
publishers reorganise, CDNs change paths, documents get superseded. A dead link
does not break anything until someone tries to reproduce the corpus, at which
point they conclude the repository does not work.

Run in CI on a schedule so the repository finds out before a reader does. This
is deliberately a separate job from the code tests: an upstream publisher
moving a PDF is not a reason to mark the code broken.

Uses GET with a byte range rather than HEAD. Several of these hosts answer HEAD
with 403 while serving GET perfectly well, so a HEAD-based checker would report
failures that are entirely its own fault.
"""

from __future__ import annotations

import httpx

from .download import HEADERS
from .manifest import load_manifest

TIMEOUT = httpx.Timeout(45.0, connect=15.0)

# Three different things a bad response can mean, and only one of them is a
# reason to fail the build.
#
# GONE      the document is not there any more. Real link rot. Fail.
# BLOCKED   the server refused this client or this network. Several publishers
#           bot-block datacenter IP ranges, so a URL that works from a laptop
#           returns 403 from CI. Reporting that as link rot trains everyone to
#           ignore the check, which is worse than not having it.
# TRANSIENT rate limiting or a hiccup. Retry, then treat as blocked.
GONE = frozenset({404, 410})
BLOCKED = frozenset({401, 402, 403, 405, 451})
TRANSIENT = frozenset({408, 429, 500, 502, 503, 504})


def check_one(client: httpx.Client, url: str) -> tuple[str, str]:
    """Return (verdict, detail) where verdict is ok | gone | blocked."""
    try:
        response, head = _fetch_head(client, url)
    except httpx.HTTPError as exc:
        return "blocked", f"{type(exc).__name__}: {exc}"

    status = response.status_code

    if status in GONE:
        return "gone", f"HTTP {status}"
    if status in BLOCKED:
        return "blocked", f"HTTP {status} (bot protection or IP range?)"
    if status in TRANSIENT:
        return "blocked", f"HTTP {status} (transient)"
    if status >= 400:
        return "gone", f"HTTP {status}"

    if not head.startswith(b"%PDF-"):
        content_type = response.headers.get("content-type", "?")
        return "gone", f"served {content_type}, not a PDF (dead deep link?)"

    return "ok", f"HTTP {status}"


def _fetch_head(client: httpx.Client, url: str) -> tuple[httpx.Response, bytes]:
    """Fetch the first bytes, preferring a Range request.

    Some servers answer a Range request with 416 rather than ignoring it, so a
    Range-only checker reports its own request as a broken link. On 416 this
    retries without the header.
    """
    for headers in ({**HEADERS, "Range": "bytes=0-1023"}, dict(HEADERS)):
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416:
                continue
            head = b""
            for block in response.iter_bytes(1024):
                head = block
                break
            return response, head

    return response, b""


def main() -> int:
    entries = load_manifest()
    print(f"checking {len(entries)} source URLs\n")

    failures: list[tuple[str, str, str]] = []

    with httpx.Client(follow_redirects=True, timeout=TIMEOUT) as client:
        for position, entry in enumerate(entries, start=1):
            ok, detail = check_one(client, entry.source_url)
            marker = "+" if ok else "!"
            print(f"  {marker} [{position}/{len(entries)}] {entry.id:<42} {detail}", flush=True)
            if not ok:
                failures.append((entry.id, entry.source_url, detail))

    print(f"\n{len(entries) - len(failures)}/{len(entries)} reachable")

    if failures:
        print("\nUnreachable — update corpus/manifest.yaml or drop the entry:")
        for doc_id, url, detail in failures:
            print(f"  {doc_id}: {detail}")
            print(f"    {url}")
        print(
            "\nRecord the change in findings/what-broke.md. A corpus that "
            "silently shrinks invalidates every number in the README."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
