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
# Statuses that mean "try again", not "gone".
TRANSIENT = frozenset({408, 429, 500, 502, 503, 504})


def check_one(client: httpx.Client, url: str) -> tuple[bool, str]:
    """Fetch the first bytes and confirm they look like a PDF."""
    try:
        with client.stream("GET", url, headers={**HEADERS, "Range": "bytes=0-1023"}) as response:
            status = response.status_code
            if status in TRANSIENT:
                return False, f"HTTP {status} (transient — may be rate limiting)"
            if status >= 400:
                return False, f"HTTP {status}"

            head = b""
            for block in response.iter_bytes(1024):
                head = block
                break

        if not head.startswith(b"%PDF-"):
            content_type = response.headers.get("content-type", "?")
            return False, f"served {content_type}, not a PDF (dead deep link?)"

        return True, f"HTTP {status}"
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"


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
