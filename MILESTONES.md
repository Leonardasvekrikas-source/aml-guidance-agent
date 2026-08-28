# Milestones

Six milestones, roughly a day each. Two weeks of evenings, or a hard week if you
have full days.

**The rule that matters:** every milestone ends in something you could link to a
recruiter that day. If you stop after M1 you have a retrieval benchmark over
regulatory guidance, which is already more than most portfolio RAG projects ever
become. There is no milestone whose value depends on reaching the next one.

Commit at every milestone. Push at every milestone. A branch that lives four days
is a branch that dies.

---

## M0 — Corpus and the box it runs in

**Build:** `docker-compose.yml` with Postgres 16 + pgvector and an app service.
`corpus/manifest.yaml` listing 20–30 public documents with source URL, publisher,
publication date, retrieval date. A download script that reads the manifest. The
chunks table: id, document_id, page, section heading, text, embedding vector,
created_at.

**New fundamentals:** what Docker Compose actually does — images vs containers,
volumes and why database data survives a restart, service networking and why the
app reaches Postgres by service name and not localhost. Schema design: why chunk
provenance is columns and not a JSON blob.

**Done when:** `docker compose up` gives you a Postgres you can connect to, the
schema is created by a migration file rather than by hand, and killing the
containers and bringing them back does not lose data.

**Publishable as:** "containerised Postgres + pgvector corpus store for AML
regulatory guidance, with full source provenance."

**Grill yourself:** why is chunk provenance worth storing per-chunk? What breaks
if you drop the page number? Why a migration file and not a setup script?

---

## M1 — The evaluation set, then retrieval

**Do these in this order.** Writing the questions after seeing what retrieval
returns produces an eval set that flatters the system. This is the single most
common way portfolio RAG projects fool their own authors.

**Build:** 40 questions in `eval/questions.yaml`. Thirty answerable, each with the
gold chunk ids that ought to be retrieved. Ten unanswerable by construction —
plausible AML questions the corpus genuinely does not cover. Then three retrieval
implementations: BM25 over chunks, dense over pgvector, hybrid via RRF. Then
recall@5, recall@10 and MRR for each, written to `results/retrieval.json`.

**New fundamentals:** what an embedding is and what cosine similarity measures.
Chunking as a decision with consequences — try two chunk sizes and report both.
Why recall matters more than precision at the retrieval stage when a reranker or
an LLM sits downstream.

**Predict before you run:** write down in `NOTES.md` which of the three you expect
to win and by how much, before you look. Then explain the gap between prediction
and result. This habit is the difference between running experiments and watching
them.

**Done when:** `make eval-retrieval` regenerates the table in the README, and you
can explain each number.

**Publishable as:** "a hand-authored retrieval benchmark over AML regulatory
guidance comparing lexical, dense and hybrid retrieval." This alone is a strong
artifact. Do not rush past it.

**Grill yourself:** where does BM25 beat dense retrieval, specifically, and why?
Show a question where it does. What does RRF actually do to the two ranked lists?

---

## M2 — The agent loop

**Build:** a loop with one tool, `search(query, k)`. The model may call it
repeatedly, reformulate a failing query, and must terminate in either an answer
with citations or an explicit refusal. Hard cap on iterations. Every step logged —
query issued, chunks returned, decision taken — to `results/traces/`.

**New fundamentals:** tool calling as a protocol rather than magic — the model
emits a structured request, your code executes it, the result goes back as a
message. Loop termination and why an unbounded agent is a billing incident.

**Done when:** you can read a trace and see the agent reformulate a bad query and
recover, and you can point to the line of your code that stops it looping forever.

**Publishable as:** "retrieval agent with query reformulation and bounded
iteration, fully traced."

**Grill yourself:** what does the agent do that a single retrieve-then-answer call
cannot? Show a question where the extra hop earns its cost, and one where it
doesn't. What's your cost and latency per question?

---

## M3 — Validation

This is the milestone that makes the project yours rather than a tutorial.

**Build:** the drafting step emits claims with citations in a structured format.
Validation then checks, in code, that each cited chunk was actually retrieved in
this run and that the claim is supported by its text. Unsupported claims reject the
draft and trigger one retry with the failure fed back. A second rejection returns a
refusal.

**New fundamentals:** structured output and why parsing prose is a losing game.
The distinction between the model choosing words and the pipeline owning facts.

**Done when:** you can force a failure — hand it a question adjacent to the corpus
but not in it — and watch validation reject a confident draft.

**Publishable as:** "citation validation layer that rejects unsupported claims
before they reach the user."

**Grill yourself:** what can this validator not catch? Name at least two failure
modes it passes. Why is that acceptable, or what would you add?

---

## M4 — Answer evaluation without labels

**Build:** groundedness scoring by LLM judge over the 30 answerable questions.
Citation validity, computed in code. Refusal accuracy over the 10 unanswerable
ones. Then sample 20 judge decisions, grade them yourself, and report the
disagreement rate in `results/judge_agreement.md`.

**New fundamentals:** LLM-as-judge and why an unaudited judge is just a second
model's opinion presented as a metric. The refusal set as the cheapest
hallucination detector you will ever build.

**Done when:** the README answer-quality table is populated, and the judge
agreement number is in the repo whether or not it is flattering.

**Publishable as:** "evaluation harness for a production RAG system without
labelled data, including judge validation." This is the section that gets read by
anyone hiring for retrieval work.

**Grill yourself:** what does groundedness fail to measure? Could a system score
100% and still be useless? Where did the judge disagree with you, and who was
right?

---

## M5 — Ship it

**Build:** FastAPI with one `/ask` endpoint returning answer, citations and trace
id. Dockerfile. `make` targets that actually work from a cold clone. README numbers
filled in. `findings/what-broke.md` written honestly. Limitations section current.

**Then the stranger test:** clone the repo into a fresh directory as if you had
never seen it, follow only the README, and time yourself. Every place you had to
consult your own memory is a README bug. Fix them.

**Done when:** a stranger with Docker and an API key gets an answer in under ten
minutes from `git clone`.

**Grill yourself:** what happens when Postgres is down? When the API key is
missing? When the question is in Lithuanian? You don't have to handle all of them —
you have to know which you chose not to.

---

## If you only get three days

M0, M1, and a crude retrieve-then-answer step with no agent loop. Publish that,
with the README honest about where it stops. A finished small thing beats an
abandoned large one, and this project is designed so that cut is clean.
