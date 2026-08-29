"""Download every document in the manifest.

Documents are not committed to the repository — the manifest and this script
are, which is what makes the corpus reproducible without redistributing other
people's PDFs. Each file's sha256 is recorded so that a publisher silently
replacing a document is visible rather than quietly shifting the evaluation
set underneath the benchmark.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ..config import settings
from .manifest import ManifestEntry, load_manifest

# Some publishers sit behind bot management that refuses a request lacking the
# fetch-metadata headers a browser always sends. FATF's CDN is one: without
# these it answers 403 no matter what User-Agent is supplied, and with them it
# serves the PDF to a client that identifies itself honestly as a script.
#
# The User-Agent below is deliberately truthful. Impersonating Chrome also
# works, but these documents are published for free public download and the
# honest header is enough, so there is no reason to lie about what is making
# the request.
HEADERS = {
    "User-Agent": (
        "aml-guidance-agent/0.1 (+https://github.com/Leonardasvekrikas-source/aml-guidance-agent)"
    ),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

PDF_MAGIC = b"%PDF-"

# Statuses that mean "try again", not "gone".
TRANSIENT_STATUS = frozenset({403, 408, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 3.0


@dataclass
class DownloadResult:
    entry: ManifestEntry
    path: Path | None
    sha256: str | None
    bytes_written: int
    status: str  # downloaded | cached | failed
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"downloaded", "cached"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_via_curl(url: str, destination: Path, timeout: int = 180) -> tuple[bool, str]:
    """Fetch with curl instead of httpx.

    Some publishers sit behind bot management that fingerprints the TLS
    handshake rather than reading the User-Agent header. FATF's CDN is one:
    it answers httpx with 403 no matter what headers are sent, and answers
    curl with 200 for the identical URL and identical User-Agent. That is not
    a dead link and it is not something a header can fix, so the fallback is a
    different HTTP client rather than a different request.

    curl is declared in the Dockerfile for exactly this reason.
    """
    curl = shutil.which("curl")
    if not curl:
        return False, "curl not available for fallback"

    try:
        completed = subprocess.run(
            [
                curl,
                "--silent",
                "--show-error",
                "--location",
                "--fail",
                "--max-time",
                str(timeout),
                "--retry",
                "2",
                "--retry-delay",
                "3",
                "--user-agent",
                HEADERS["User-Agent"],
                *[
                    arg
                    for key, value in HEADERS.items()
                    if key != "User-Agent"
                    for arg in ("--header", f"{key}: {value}")
                ],
                "--output",
                str(destination),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 30,
        )
    except subprocess.TimeoutExpired:
        return False, "curl timed out"

    if completed.returncode != 0:
        destination.unlink(missing_ok=True)
        detail = (completed.stderr or "").strip().splitlines()
        return False, f"curl exit {completed.returncode}: {detail[-1] if detail else 'no output'}"

    return True, "via curl"


def _fetch_with_httpx(url: str, destination: Path, client: httpx.Client) -> tuple[bool, int, str]:
    """Try httpx, retrying statuses that mean "try again" rather than "gone".

    Returns (ok, bytes_written, detail).
    """
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with client.stream("GET", url) as response:
                if response.status_code in TRANSIENT_STATUS:
                    last_error = f"HTTP {response.status_code}"
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(BACKOFF_SECONDS * attempt)
                        continue
                    return False, 0, last_error

                response.raise_for_status()

                written = 0
                with destination.open("wb") as handle:
                    for block in response.iter_bytes(1 << 16):
                        handle.write(block)
                        written += len(block)
                return True, written, ""

        except httpx.HTTPStatusError as exc:
            # A non-transient status. The document is genuinely not there.
            return False, 0, f"HTTP {exc.response.status_code}"
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == MAX_ATTEMPTS:
                return False, 0, last_error
            time.sleep(BACKOFF_SECONDS * attempt)

    return False, 0, last_error or "exhausted attempts"


def download_one(
    entry: ManifestEntry,
    client: httpx.Client,
    force: bool = False,
) -> DownloadResult:
    destination = entry.raw_path()

    if destination.exists() and not force:
        return DownloadResult(
            entry=entry,
            path=destination,
            sha256=sha256_of(destination),
            bytes_written=destination.stat().st_size,
            status="cached",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name and move into place, so an interrupted
    # download cannot leave a truncated PDF that later looks cached.
    temporary = destination.with_suffix(".part")

    ok, written, detail = _fetch_with_httpx(entry.source_url, temporary, client)

    # Bot management that fingerprints the TLS handshake refuses httpx and
    # accepts curl for the same URL with the same headers. That is a client
    # problem, not a dead link, so a refusal is worth one attempt with a
    # different client before the document is written off.
    note = ""
    if not ok:
        temporary.unlink(missing_ok=True)
        curl_ok, curl_detail = _download_via_curl(entry.source_url, temporary)
        if not curl_ok:
            temporary.unlink(missing_ok=True)
            return DownloadResult(
                entry, None, None, 0, "failed", f"{detail}; fallback {curl_detail}"
            )
        written = temporary.stat().st_size
        note = f"{detail} -> recovered via curl"

    # A 200 response is not proof of a PDF. Several publishers answer a dead
    # deep link with an HTML search page, which would otherwise be ingested as
    # a document full of navigation text.
    with temporary.open("rb") as handle:
        head = handle.read(len(PDF_MAGIC))
    if head != PDF_MAGIC:
        temporary.unlink(missing_ok=True)
        return DownloadResult(
            entry,
            None,
            None,
            written,
            "failed",
            "response was not a PDF (probably an HTML error or search page)",
        )

    temporary.replace(destination)
    return DownloadResult(
        entry=entry,
        path=destination,
        sha256=sha256_of(destination),
        bytes_written=written,
        status="downloaded",
        detail=note,
    )


def download_all(force: bool = False) -> list[DownloadResult]:
    entries = load_manifest()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []

    with httpx.Client(
        headers=HEADERS,
        follow_redirects=True,
        timeout=httpx.Timeout(60.0, connect=20.0),
    ) as client:
        for position, entry in enumerate(entries, start=1):
            result = download_one(entry, client, force=force)
            results.append(result)
            size = f"{result.bytes_written / 1_048_576:.1f}MB" if result.bytes_written else "-"
            marker = {"downloaded": "+", "cached": "=", "failed": "!"}[result.status]
            line = f"  {marker} [{position}/{len(entries)}] {entry.id:<40} {size:>8}"
            if result.detail:
                line += f"  {result.detail}"
            print(line, flush=True)

    return results


def main() -> int:
    print(f"retrieval started {datetime.now(UTC).isoformat(timespec='seconds')}")
    results = download_all(force="--force" in sys.argv)

    failed = [r for r in results if not r.ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} available, {len(failed)} failed")
    if failed:
        print("\nfailed documents — fix the URL in corpus/manifest.yaml or drop the entry:")
        for result in failed:
            print(f"  {result.entry.id}: {result.detail}")
            print(f"    {result.entry.source_url}")
        # A partial corpus is a legitimate state to work in, but it must not be
        # mistaken for a complete one, so this exits non-zero.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
