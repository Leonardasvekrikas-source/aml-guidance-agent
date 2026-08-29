"""Ask one question from the command line.

make ask Q="What indicators suggest trade-based money laundering?"
.\\run.ps1 ask "What indicators suggest trade-based money laundering?"
"""

from __future__ import annotations

import sys

from ..config import DEFAULT_PROFILE
from ..retrieval import build_retriever
from .pipeline import Pipeline, write_trace


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('usage: python -m aml_agent.agent.cli "your question"')
        return 2

    retriever = build_retriever("hybrid", DEFAULT_PROFILE)
    result = Pipeline(retriever).ask(question)
    write_trace(result)

    print(f"\n{'=' * 72}")
    print(f"Q: {question}")
    print(f"{'=' * 72}\n")

    if result.outcome == "answered":
        print(result.summary)
        print("\nSources:")
        for citation in result.citations:
            page = f"p.{citation['page']}" if citation["page"] else "page unknown"
            print(f"  - {citation['title']} ({citation['publisher']}), {page}")
            print(f"    {citation['source_url']}")
    elif result.outcome == "refused":
        print("INSUFFICIENT EVIDENCE\n")
        print(result.refusal_reason)
    else:
        print(f"ERROR: {result.refusal_reason}")

    validation = result.validation
    print(
        f"\n[trace {result.trace_id}] attempts={result.attempts} "
        f"searches={result.total_searches} "
        f"tokens_in={result.total_input_tokens} tokens_out={result.total_output_tokens} "
        f"{result.latency_ms / 1000:.1f}s"
    )
    if validation:
        print(
            f"[validation] {len(validation.verdicts)} claims, "
            f"{len(validation.failures)} rejected, "
            f"groundedness {validation.groundedness():.2f}"
        )
        for verdict in validation.failures:
            print(f"  REJECTED: {verdict.claim[:80]}")
            print(f"            {verdict.detail[:140]}")

    print(f"\ntrace written to results/traces/{result.trace_id}.json")
    return 0 if result.outcome in {"answered", "refused"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
