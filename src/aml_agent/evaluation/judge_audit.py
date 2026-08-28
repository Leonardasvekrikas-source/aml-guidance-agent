"""Judge audit: sample judge decisions for human grading.

An LLM judge that nobody has checked is a second model's opinion with a decimal
point on it. This script samples its decisions, writes them out for a human to
grade, and then computes the disagreement rate.

Two commands:

    make judge-audit           # sample decisions into results/judge_agreement.md
    make judge-audit-score     # after grading, compute the disagreement rate

The grading file is deliberately committed whether the number is flattering or
not. A judge-agreement figure that only appears when it is good is not a
measurement.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from ..config import settings

SAMPLE_SIZE = 20

HEADER = """# Judge agreement audit

The groundedness figure in the README comes from an LLM judge. This file is the
check on that judge.

**How to complete it.** For each case below, read the answer and the passages,
then write `agree` or `disagree` on the `Human:` line — `agree` meaning you
reach the same verdict the judge did, `disagree` meaning you do not. Add a
sentence saying why when you disagree.

Grade before looking at the judge's reasoning if you can. Reading its
justification first makes agreement more likely and the audit worth less.

Then run `make judge-audit-score`.

---

"""


def _pick(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Deterministic stratified sample.

    Not random: the sample must be reproducible from committed artifacts, and
    it must include both verdicts. Sampling only the cases the judge called
    ungrounded would measure one error direction and miss false "grounded"
    verdicts entirely, which are the dangerous ones.
    """
    judged = [r for r in rows if r.get("judge") and not r["judge"].get("error")]
    grounded = [r for r in judged if r["judge"]["grounded"]]
    ungrounded = [r for r in judged if not r["judge"]["grounded"]]

    grounded.sort(key=lambda r: r["question_id"])
    ungrounded.sort(key=lambda r: r["question_id"])

    # Take every ungrounded case first — they are rarer and more informative —
    # then fill the rest with grounded ones spread evenly across the set.
    picked = ungrounded[:size]
    remaining = size - len(picked)
    if remaining > 0 and grounded:
        step = max(1, len(grounded) // remaining)
        picked.extend(grounded[::step][:remaining])

    return picked[:size]


def write_audit() -> int:
    answers_path = settings.results_dir / "answers.json"
    if not answers_path.exists():
        print("no results/answers.json — run `make eval-answers` first")
        return 1

    data = json.loads(answers_path.read_text(encoding="utf-8"))
    sample = _pick(data["per_question"], SAMPLE_SIZE)
    if not sample:
        print("no judged answers to audit")
        return 1

    blocks = [HEADER]
    for position, row in enumerate(sample, start=1):
        judge = row["judge"]
        verdict = "GROUNDED" if judge["grounded"] else "NOT GROUNDED"
        trace_path = settings.traces_dir / f"{row['trace_id']}.json"
        answer = ""
        if trace_path.exists():
            answer = json.loads(trace_path.read_text(encoding="utf-8")).get("summary", "")

        blocks.append(
            f"## Case {position} — `{row['question_id']}`\n\n"
            f"**Question.** {row['question']}\n\n"
            f"**Answer given.**\n\n> {answer or '(see trace)'}\n\n"
            f"**Passages.** `results/traces/{row['trace_id']}.json`\n\n"
            f"**Judge verdict.** {verdict}\n\n"
            f"<details><summary>Judge reasoning (read after grading)</summary>\n\n"
            f"{judge['reason']}\n\n"
            + (
                "Unsupported assertions named:\n"
                + "\n".join(f"- {a}" for a in judge["unsupported_assertions"])
                + "\n\n"
                if judge["unsupported_assertions"]
                else ""
            )
            + "</details>\n\n"
            f"**Human:** \n\n"
            f"**Note:** \n\n---\n"
        )

    output = settings.results_dir / "judge_agreement.md"
    output.write_text("\n".join(blocks), encoding="utf-8", newline="\n")
    print(f"wrote {len(sample)} cases to {output}")
    print("Grade each 'Human:' line as agree or disagree, then `make judge-audit-score`.")
    return 0


def score_audit() -> int:
    path = settings.results_dir / "judge_agreement.md"
    if not path.exists():
        print("no results/judge_agreement.md — run `make judge-audit` first")
        return 1

    text = path.read_text(encoding="utf-8")
    verdicts = [m.strip().lower() for m in re.findall(r"\*\*Human:\*\*\s*(\w+)?", text)]
    graded = [v for v in verdicts if v in {"agree", "disagree"}]

    if not graded:
        print(
            f"none of the {len(verdicts)} cases have been graded yet. "
            "Write agree or disagree on each 'Human:' line."
        )
        return 1

    disagreements = sum(1 for v in graded if v == "disagree")
    rate = disagreements / len(graded)

    summary = (
        f"\n---\n\n## Result\n\n"
        f"- Cases graded: **{len(graded)}** of {len(verdicts)}\n"
        f"- Disagreements: **{disagreements}**\n"
        f"- Judge–human disagreement rate: **{rate:.1%}**\n\n"
        f"This is the error bar on the groundedness figure in the README. "
        f"A groundedness score is only as trustworthy as this number is small.\n"
    )

    if "## Result" in text:
        text = text.split("\n---\n\n## Result")[0]
    path.write_text(text + summary, encoding="utf-8", newline="\n")

    print(f"graded {len(graded)}/{len(verdicts)} — disagreement rate {rate:.1%}")
    print(f"appended the result to {path}")
    return 0


def main() -> int:
    if "--score" in sys.argv:
        return score_audit()
    return write_audit()


if __name__ == "__main__":
    raise SystemExit(main())
