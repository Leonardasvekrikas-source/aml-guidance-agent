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
from typing import Any, cast

import anthropic

from ..config import settings
from ..cost import usd
from ..llm import make_client
from ..retrieval.base import Hit, Retriever
from .tools import SEARCH_TOOL, run_search

# Raised from 6 after measurement: the agent issues one to two searches per
# iteration, so a question needing several reformulations plus a final
# answer could hit the cap while still searching, and be recorded as a
# refusal it never actually made. The cap exists to bound cost, not to
# decide questions.
MAX_ITERATIONS = 10
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

# --- prompt caching -------------------------------------------------------
#
# By the eighth search this conversation carries tens of thousands of tokens,
# and almost all of it is byte-identical to the previous request: the same
# tools, the same system prompt, and every earlier search result. Without
# caching that prefix is paid for again on every turn, which is most of the
# cost of a question.
#
# The prefix is rendered tools -> system -> messages, so a breakpoint is placed
# at the end of the system prompt (everything before it is static and cached
# for the whole run) and another is moved forward onto the newest tool results
# each turn, so the previous turn's write becomes this turn's read.
#
# Caching is a PREFIX match: any byte change before a breakpoint invalidates
# everything after it. That is why the system prompt is a module constant and
# the tool list is never reordered.
CACHED_SYSTEM = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


def _move_cache_breakpoint(messages: list[Any]) -> None:
    """Keep exactly one moving breakpoint, on the latest content.

    The API allows at most four breakpoints, so old ones are cleared as the
    conversation grows rather than accumulating until a request is rejected.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)

    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, list) and content and isinstance(content[-1], dict):
            content[-1]["cache_control"] = {"type": "ephemeral"}
            return


@dataclass
class Claim:
    text: str
    chunk_ids: list[int]


def _parse_claims(raw: Any) -> list[Claim]:
    """Turn the model's `claims` argument into Claim objects, defensively.

    The tool schema says each claim is an object with `text` and `chunk_ids`,
    and the model mostly complies — but not always. Observed in a live run: a
    bare string in place of the object, which crashed the loop mid-evaluation
    and cost a partial run.

    A malformed claim is kept rather than dropped, with no citations. That is
    deliberate: an uncitable claim then fails the provenance check and rejects
    the draft, which is the correct outcome. Discarding it would silently
    shrink the answer and let the rest pass validation as though the model had
    never made the assertion.
    """
    claims: list[Claim] = []
    for item in raw or []:
        if isinstance(item, dict):
            chunk_ids: list[int] = []
            for value in item.get("chunk_ids") or []:
                try:
                    chunk_ids.append(int(value))
                except (TypeError, ValueError):
                    continue
            claims.append(Claim(text=str(item.get("text", "")), chunk_ids=chunk_ids))
        else:
            claims.append(Claim(text=str(item), chunk_ids=[]))
    return claims


@dataclass
class AgentResult:
    question: str
    trace_id: str
    outcome: str  # answered | refused | exhausted | error
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
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    # The conversation as it stood when the loop finished. Kept in memory
    # so a rejected draft can be corrected in place rather than restarted;
    # deliberately not written to the trace, where it would duplicate every
    # search result already recorded step by step.
    messages: list[Any] = field(default_factory=list)
    error: str = ""

    @property
    def usd(self) -> float:
        return usd(
            self.model,
            self.input_tokens,
            self.output_tokens,
            self.cache_write_tokens,
            self.cache_read_tokens,
        )

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
            "cache_read_tokens": self.cache_read_tokens,
            "usd": round(self.usd, 5),
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

    def run(self, question: str, resume: AgentResult | None = None) -> AgentResult:
        """Answer a question, optionally continuing a previous attempt.

        ``resume`` carries the conversation from a draft that validation
        rejected. Continuing it matters for both correctness and cost: a
        restarted loop throws away every passage it already retrieved and has
        to find them again, which in practice exhausted the iteration budget
        re-searching for evidence it had already seen. Continuing means the
        agent only has to fix the claim it was told about.
        """
        started = time.perf_counter()
        result = AgentResult(
            question=question,
            trace_id=uuid.uuid4().hex[:12],
            outcome="error",
            model=self.model,
        )
        messages: list[Any] = [{"role": "user", "content": question}]

        finished = False
        while not finished:
            # --- termination guard ------------------------------------------
            # This is the line that stops the loop running forever. It is
            # checked before every API call, so no path through the loop can
            # skip it.
            if result.iterations >= self.max_iterations:
                result.outcome = "exhausted"
                result.refusal_reason = (
                    f"Reached the {self.max_iterations}-iteration cap without reaching an answer."
                )
                result.trace.append({"step": "halt", "reason": "iteration cap reached"})
                break

            result.iterations += 1
            _move_cache_breakpoint(messages)

            try:
                # Tool and message payloads are plain dicts matching the
                # documented wire format; the SDK types them as TypedDicts.
                # Prompt caching. Each iteration re-sends every previous
                # search result, so by the eighth search the conversation is
                # tens of thousands of tokens and most of it is byte-identical
                # to the previous request. Caching turns that repeated prefix
                # into cache reads at a tenth of the input price.
                #
                # The prefix order is tools -> system -> messages, and all
                # three are stable here: the tool list never changes and the
                # system prompt is a constant, so the cache breakpoint moves
                # forward with the conversation instead of being invalidated
                # by it.
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    system=CACHED_SYSTEM,  # type: ignore[arg-type]
                    tools=TOOLS,  # type: ignore[arg-type]  # plain dicts, not SDK TypedDicts
                    messages=messages,
                )
            except anthropic.APIError as exc:
                result.outcome = "error"
                result.error = f"{type(exc).__name__}: {exc}"
                result.trace.append({"step": "api_error", "detail": result.error})
                break

            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens
            # Cache fields are absent on responses that used no caching.
            result.cache_write_tokens += (
                getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            )
            result.cache_read_tokens += getattr(response.usage, "cache_read_input_tokens", 0) or 0

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
                # The SDK types tool input as `object` because its shape is
                # defined by the schema we supplied, which it cannot see.
                # Casting here states that assumption once, at the boundary,
                # rather than at every field access below.
                payload = cast(dict[str, Any], block.input)

                if block.name == "search":
                    rendered, hits = run_search(self.retriever, payload)
                    result.searches += 1
                    for hit in hits:
                        result.retrieved_chunk_ids.add(hit.chunk_id)
                        result.retrieved_hits[hit.chunk_id] = hit
                    result.trace.append(
                        {
                            "step": "search",
                            "iteration": result.iterations,
                            "query": payload.get("query"),
                            "k": payload.get("k", 5),
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
                    result.claims = _parse_claims(payload.get("claims"))
                    result.summary = str(payload.get("summary", ""))
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
                    result.refusal_reason = str(payload.get("reason", ""))
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

        result.messages = messages
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result
