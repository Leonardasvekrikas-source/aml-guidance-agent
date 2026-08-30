"""What a run costs, in dollars.

A retrieval agent that issues several model calls per question has a cost per
question, and "it works" is not an answer to what that is. This turns the token
counts the API already returns into money, so the figure in the README is
measured rather than estimated from a blog post.

Prices are per million tokens and are hardcoded on purpose: a cost figure that
silently changes when a price does is not reproducible. If a price moves, this
table is the one place to change, and the results files record which table
produced them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float

    # Cache writes cost 1.25x input, cache reads 0.1x. Both are per the
    # published pricing model rather than measured here.
    @property
    def cache_write(self) -> float:
        return self.input * 1.25

    @property
    def cache_read(self) -> float:
        return self.input * 0.10


PRICES: dict[str, Price] = {
    "claude-opus-5": Price(input=5.00, output=25.00),
    "claude-sonnet-5": Price(input=2.00, output=10.00),
    "claude-haiku-4-5": Price(input=1.00, output=5.00),
}

# Used when a model is not in the table, so an unknown model produces a visibly
# wrong number rather than a silent zero that looks like "free".
UNKNOWN = Price(input=float("nan"), output=float("nan"))


def price_for(model: str) -> Price:
    return PRICES.get(model, UNKNOWN)


def usd(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    p = price_for(model)
    return (
        input_tokens * p.input
        + output_tokens * p.output
        + cache_write_tokens * p.cache_write
        + cache_read_tokens * p.cache_read
    ) / 1_000_000


@dataclass
class Spend:
    """Running total across a batch of calls."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def add(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_write_tokens += cache_write_tokens
        self.cache_read_tokens += cache_read_tokens
        self.calls += 1

    @property
    def total_usd(self) -> float:
        return usd(
            self.model,
            self.input_tokens,
            self.output_tokens,
            self.cache_write_tokens,
            self.cache_read_tokens,
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "usd": round(self.total_usd, 4),
        }
