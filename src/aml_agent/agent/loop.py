"""The agent loop.

The whole contract in one place:

  * The model may call ``search`` as often as it likes, reformulating when a
    query returns nothing useful.
  * It must finish by calling either ``answer`` or ``refuse``. Those are tools
    rather than free prose because a structured answer can be validated and a
    paragraph cannot.
  * The loop is bounded. An agent that can loop forever is not a clever agent,
    it is an unbounded bill and an unbounded latency.

Termination is deliberately explicit. Every step is recorded in the trace,
including the ones that failed, because a trace that only shows successes
cannot be used to debug anything.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import anthropic

from ..config import settings
from ..llm import make_client
from ..retrieval.base import Hit, Retriever
from .tools import SEARCH_TOOL, run_search

MAX_ITERATIONS = 6
MAX_TOKENS = 8000

SYSTEM_PROMPT = """You answer questions about anti-money-laundering typologies \
and regulatory expectations, using ONLY a corpus of public AML guidance \
documents that you search with the search tool.

How to work:

1. Search before you answer. Never answer from your own knowledge of AML, even
   when you are confident. Your knowledge is not the corpus, and a claim you
   cannot cite is worthless here.
2. If a search returns nothing useful, reformulate and search again. Use the
   regulatory vocabulary the documents would actually use rather than the
   phrasing of the question. Two or three well-chosen searches beat one long
   one.
3. When you have enough evidence, call the answer tool. Break the answer into
   discrete claims, and cite the chunk ids that support each claim. Cite only
   chunk ids you actually saw in a search result in THIS conversation.
4. If the corpus does not support an answer, call the refuse tool. Refusing is
   a correct outcome, not a failure. A confident answer the corpus does not
   support is far worse than admitting the corpus does not cover it.

Cite precisely. Each claim must be supported by the specific chunk you cite,
not by a chunk that merely discusses the same topic."""

ANSWER_TOOL: dict[str, Any] = {
    "name": "answer",
    "description": (
        "Provide the final answer, broken into claims, each citing the chunk ids "
        "that support it. Call this only when the retrieved passages genuinely "
        "support what you are asserting."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "description": "The answer as discrete, individually checkable claims.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "One factual claim, stated plainly.",
                        },
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Chunk ids supporting THIS claim. Must be ids seen "
                                "in a search result in this conversation."
                            ),
                        },
                    },
                    "required": ["text", "chunk_ids"],
                    "additionalProperties": False,
                },
            },
            "summary": {
                "type": "string",
                "description": "A short prose answer to the question, for a human reader.",
            },
        },
        "required": ["claims", "summary"],
        "additionalProperties": False,
    },
}

REFUSE_TOOL: dict[str, Any] = {
    "name": "refuse",
    "description": (
        "Decline to answer because the corpus does not contain the information. "
        "Use this rather than assembling an answer from loosely related passages."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "What was searched for, and what was missing.",
            }
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
}

TOOLS = [SEARCH_TOOL, ANSWER_TOOL, REFUSE_TOOL]


@dataclass
class Claim:
    text: str
    chunk_ids: list[int]


@dataclass
class AgentResult:
    question: str
    trace_id: str
    outcome: str                      # answered | refused | exhausted | error
    summary: str = ""
    claims: list[Claim] = field(default_factory=list)
    refusal_reason: str = ""
    iterations: int = 0
    searches: int = 0
    retrieved_chunk_ids: set[int] = field(default_factory=set)
    retrieved_hits: dict[int, Hit] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""

    @property
    def answered(self) -> bool:
        return self.outcome == "answered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "outcome": self.outcome,
            "summary": self.summary,
            "claims": [{"text": c.text, "chunk_ids": c.chunk_ids} for c in self.claims],
            "refusal_reason": self.refusal_reason,
            "iterations": self.iterations,
            "searches": self.searches,
            "retrieved_chunk_ids": sorted(self.retrieved_chunk_ids),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
            "trace": self.trace,
        }


class AgentLoop:
    def __init__(
        self,
        retriever: Retriever,
        client: anthropic.Anthropic | None = None,
        model: str | None = None,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.retriever = retriever
        self.client = client or make_client()
        self.model = model or settings.anthropic_model
        self.max_iterations = max_iterations

    def run(self, question: str) -> AgentResult:
        started = time.perf_counter()
        result = AgentResult(
            question=question,
            trace_id=uuid.uuid4().hex[:12],
            outcome="error",
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

        finished = False
        while not finished:
            # --- termination guard ------------------------------------------
            # This is the line that stops the loop running forever. It is
            # checked before every API call, so no path through the loop can
            # skip it.
            if result.iterations >= self.max_iterations:
                result.outcome = "exhausted"
                result.refusal_reason = (
                    f"Reached the {self.max_iterations}-iteration cap without "
                    "reaching an answer."
                )
                result.trace.append({"step": "halt", "reason": "iteration cap reached"})
                break

            result.iterations += 1

            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )
            except anthropic.APIError as exc:
                result.outcome = "error"
                result.error = f"{type(exc).__name__}: {exc}"
                result.trace.append({"step": "api_error", "detail": result.error})
                break

            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens

            # A safety refusal is a different thing from the agent choosing to
            # refuse because the corpus is silent. Conflating them would
            # corrupt the M4 refusal metric.
            if response.stop_reason == "refusal":
                result.outcome = "error"
                result.error = "model safety refusal"
                result.trace.append({"step": "safety_refusal"})
                break

            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                # The model replied in prose instead of calling a tool. Push it
                # back rather than accepting an uncitable answer.
                said = " ".join(b.text for b in response.content if b.type == "text")
                result.trace.append({"step": "prose_without_tool", "text": said[:400]})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Do not answer in prose. Call search to gather evidence, "
                            "then call answer with claims and chunk ids, or refuse."
                        ),
                    }
                )
                continue

            # All tool results for one assistant turn must go back in a SINGLE
            # user message, or the model learns to stop issuing parallel calls.
            tool_results: list[dict[str, Any]] = []

            for block in tool_uses:
                if block.name == "search":
                    rendered, hits = run_search(self.retriever, dict(block.input))
                    result.searches += 1
                    for hit in hits:
                        result.retrieved_chunk_ids.add(hit.chunk_id)
                        result.retrieved_hits[hit.chunk_id] = hit
                    result.trace.append(
                        {
                            "step": "search",
                            "iteration": result.iterations,
                            "query": block.input.get("query"),
                            "k": block.input.get("k", 5),
                            "returned_chunk_ids": [h.chunk_id for h in hits],
                        }
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": rendered,
                        }
                    )

                elif block.name == "answer":
                    raw_claims = block.input.get("claims") or []
                    result.claims = [
                        Claim(
                            text=str(c.get("text", "")),
                            chunk_ids=[int(i) for i in (c.get("chunk_ids") or [])],
                        )
                        for c in raw_claims
                    ]
                    result.summary = str(block.input.get("summary", ""))
                    result.outcome = "answered"
                    result.trace.append(
                        {
                            "step": "answer",
                            "iteration": result.iterations,
                            "claims": len(result.claims),
                        }
                    )
                    finished = True

                elif block.name == "refuse":
                    result.refusal_reason = str(block.input.get("reason", ""))
                    result.outcome = "refused"
                    result.trace.append(
                        {
                            "step": "refuse",
                            "iteration": result.iterations,
                            "reason": result.refusal_reason[:300],
                        }
                    )
                    finished = True

                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "ERROR: unknown tool " + repr(block.name),
                            "is_error": True,
                        }
                    )

            if finished:
                break
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result
