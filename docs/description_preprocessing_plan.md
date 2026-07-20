# Goodreads Description Preprocessing — Refined Execution Plan

## Background (read this first — you won't have this context otherwise)

The pipeline crawls Goodreads' internal GraphQL API to pull book metadata, generates text
embeddings from that metadata, and feeds the embeddings into ML models that predict how a user
would rate a book. This plan is about one specific field in that metadata — the book
**description** — and a data-quality problem with it that affects the embedding step.

**Where descriptions come from.** `goodreads_ranker/ingestion/api_client.py` defines the
GraphQL query used to fetch a book; it requests a `description` field directly from Goodreads'
API, which Goodreads returns **HTML-formatted** (e.g. `<br />`, `<i>...</i>` for italics).
`goodreads_ranker/ingestion/crawler.py` flattens the raw GraphQL response
(`flatten_book()`) and saves the description **verbatim, HTML tags and all** into the
`books.description` column via `goodreads_ranker/core/db.py`. Nothing cleans it at ingestion
time — this is by design, the raw value is preserved for now.

**Where descriptions go next.** `goodreads_ranker/ml/embedder.py` is what actually consumes
descriptions. It reads book metadata (title, author, genres, description) from the DB,
does a light whitespace cleanup on the description (`re.sub(r"\s+", " ", desc).strip()` —
nothing else), concatenates it with the other fields into one input string per book, and sends
batches of these strings to a local Ollama embedding model. The resulting vectors are stored in
a `book_embeddings` table, keyed by book + embedding model, alongside an md5 hash
(`text_hash`) of the exact input string that produced them. That hash is the pipeline's
staleness check: if the input text for a book changes, its embedding is automatically
considered outdated and gets regenerated next time `python main.py embed` runs. This matters
later — it means changing how descriptions are cleaned doesn't require any special migration
step, it just needs a normal re-embed pass.

**What the embeddings are used for.** `goodreads_ranker/ml/predictor.py` loads these vectors
and feeds them into an ensemble — cosine-metric KNN, BayesianRidge, and an RBF-kernel SVR —
that predicts a rating for every book from the user's own ratings and friends' calibrated
ratings. All three model types make predictions based on the absolute **geometric position** of
an embedding in vector space (distances, neighborhoods), not just a relative ranking. This
detail is the whole reason description formatting is worth worrying about — see the next
section.

**Why this plan exists.** A rough, ad hoc script — `old/diagnose_descriptions.py` (the `old/`
folder means it isn't part of the live pipeline, it was written for one-off analysis) — was run
against the live `books.description` data and found it's inconsistently formatted:
**71% of descriptions contain raw HTML tags** (mostly `<br>`, `<i>`, `<b>`), and smaller
fractions contain HTML entities (`&amp;`), stray Unicode control/format characters
(zero-width spaces, bidi marks), encoding corruption ("mojibake" — e.g. UTF‑8 bytes that got
decoded as Latin‑1 somewhere upstream, producing garbled sequences like `â€¦`), and irregular
whitespace. None of this is stripped before the text reaches the embedding model — again,
`embedder.py`'s only cleanup is the whitespace collapse mentioned above.

**The open question this plan resolves.** Does this formatting noise matter enough to clean up
before embedding, and if so, how do we build and validate that cleanup without over-investing
in it? The "should we bother" question has already been reasoned through (see next section) —
this plan is about validating and shipping a specific, low-effort fix, not re-litigating
whether to do it.

---

## Why preprocessing is worth doing (already decided, not up for re-litigation)

- The prediction ensemble in `predictor.py` operates directly on embedding **geometry**
  (cosine-KNN, BayesianRidge, RBF-SVR), not just rank order. If HTML-tag presence
  systematically shifts where a description lands in vector space — and that shift correlates
  with anything real, like genre or publisher (an unformatted description from one source
  "looking different" in vector space from a formatted one from another source, for reasons
  that have nothing to do with the book's actual content) — the models can't tell "similar
  content" apart from "similar formatting." That's a confound-removal argument: it doesn't
  require proving the noise is large, just that it's structurally present and easy to remove.
- The fix is cheap: one small function, one dependency addition, one re-embed pass.
- Rollout is mechanical: because `embedder.py` already invalidates embeddings via `text_hash`,
  changing the cleaning logic automatically marks every affected book stale for re-embedding —
  no new migration code is needed (see Step 3).

---

## Step 1 — Enhance the diagnostic script (throwaway)

**What `old/diagnose_descriptions.py` currently does, briefly:** it loads descriptions from
either a CSV or the sqlite DB, runs `analyze_text()` over each row (checks for HTML tags/
entities via regex, Unicode control/format characters via `unicodedata.category()`, multi-space/
multi-newline patterns, leading/trailing whitespace, raw URLs, and a heuristic mojibake check
based on known garbled-character markers like `â€`), aggregates the per-row findings into
corpus-wide percentages in `aggregate()`, and prints a report (`print_report()`) with sample
offending rows and a hardcoded list of cleaning recommendations. It's a standalone script, not
imported by anything else in the pipeline.

**Lifespan:** One-time decision-support artifact. Not promoted into the `goodreads_ranker`
package, not given a stable API, not wired into tests/CI. Fine to leave sitting in `old/` or
delete once Step 2 passes.

**Bug-fix scope — only fix what could change the decision:**
- Fix: the `is_null` check in `analyze_text()` only catches `None`/NaN — a genuinely empty
  string (`""`) slips past it and gets silently counted as a normal length-0 row instead of
  a missing/empty one. This affects trust in the length stats (the original run reported
  `min=0` while also claiming 0% null/empty, which is the bug surfacing). Distinguish "null"
  from "empty" explicitly.
- Skip (cosmetic, don't change the decision — leave a one-line comment instead of fixing):
  the HTML-entity regex has false positives on patterns shaped like `Q&A;`, the tag regex only
  counts opening tags (not closing tags like `</i>`), and the URL regex over-counts URLs that
  live inside `<a href="...">` markup and will vanish automatically once tags are stripped —
  none of these change whether cleaning is worthwhile.

**New capability — HTML-tag-presence correlation check.** The original diagnostic only
measures *how much* lexical noise exists; it says nothing about whether that noise is randomly
distributed or systematically correlated with something else (e.g., certain genres or
publishers consistently having unformatted descriptions). That correlation is the real risk
described above, so this is the one substantial new piece of analysis to add:
- **Grouping variable:** a binary `has_html_tags` flag per description (from the existing
  `HTML_TAG_RE`). This is the only artifact flag with real statistical power (the ~71/29
  split found earlier). The rarer flags (control_chars, mojibake, format_chars, entities — all
  under 3% prevalence) stay in the existing "sample offending rows" eyeball format; don't
  attempt correlation stats against them, the counts are too small to be meaningful.
- **Variables checked against it:** genre, rating_count, avg_rating, publisher, language_name,
  publication_time.
  - These live in the DB schema managed by `goodreads_ranker/core/db.py`: genres are
    many-to-many via a `book_genres` join table against a `genres` table (a book can have
    several genres, there's no single "primary" genre marked in the schema); `rating_count`/
    `avg_rating` aren't stored directly but are derivable from the `star_1..star_5` count
    columns on `books` (same approach `predictor.py` already uses); `publisher`,
    `language_name`, and `publication_time` are plain columns on `books`.
- **Data plumbing:** the current script only pulls a single `description` column via
  `load_data()`. Add a dedicated query joining `books` with `book_genres`/`genres`, computing
  `rating_count`/`avg_rating` from `star_1..star_5`, and pulling `publisher`, `language_name`,
  `publication_time` directly from `books`. (Check both `publication_time` and
  `original_publication_time` for null rates and use whichever is more consistently populated.)
- **Method, reported as effect size — not p-values.** At N≈22,000 rows, standard significance
  tests (chi-square, Mann-Whitney) will read "statistically significant" for almost any
  difference, including ones too small to matter practically. Report magnitude instead:
  - *Genre* (multi-label): for the top ~20 most frequent genres (or all genres with ≥30 total
    occurrences, to avoid noise from rare tags), compute the % of tagged-group books carrying
    that genre vs. the % of untagged-group books carrying it. Report as a table ranked by
    absolute percentage-point gap.
  - *Publisher, language_name* (high-cardinality categorical — likely thousands of distinct
    publishers): bucket to the top ~15–20 most frequent values plus an "other" bucket; same
    percentage-point-gap comparison as genre.
  - *rating_count, avg_rating, publication_time* (continuous): report median/IQR per group,
    plus a rank-biserial correlation (the natural effect-size pairing with a Mann-Whitney U
    test) as the magnitude-of-difference metric — compute it, but don't lead with or gate
    anything on the p-value.
- **No hard pass/fail threshold.** This is exploratory evidence for the writeup, not a gate on
  whether to proceed with preprocessing — that decision already stands independent of what
  this finds (see "Why preprocessing is worth doing" above). Gaps in the low single digits (or
  rank-biserial correlations under ~0.1) are noise-level; anything larger is worth a sentence
  of commentary in the final writeup.

---

## Step 2 — Acceptance test (in-memory, single run)

The goal here is to validate a *candidate cleaning function* by running the (now-fixed)
diagnostic against the description corpus both before and after cleaning, in one script
execution:

- Implement the candidate `clean_description_text()` in `utils.py` first (see Step 3) so this
  test exercises the real function that will ship, not a reimplementation of it.
- Add a small runner to the (throwaway) diagnostic script that, in a single process:
  1. Pulls all raw descriptions from the DB.
  2. Applies `clean_description_text()` to every row in-memory — no DB writes, no temp
     tables/files, nothing persisted.
  3. Runs the (bug-fixed) `aggregate()` twice — once on the raw descriptions, once on the
     cleaned ones — and prints both reports back to back for a direct before/after read.
- **Structural pass criteria:** `contains_html_tags`, `contains_html_entities`,
  `contains_control_chars`, `possible_mojibake`, and `nfkc_changes_text` should all collapse to
  ~0% post-clean. Whitespace-related stats should drop substantially too (a small residual is
  fine — not all whitespace variance is a defect worth chasing).
- **Note on the correlation check from Step 1:** it does *not* get re-run post-clean. Once
  every description has its HTML tags stripped, there's no more `has_html_tags` split to group
  by — the confound is structurally eliminated by construction, not something you re-measure
  and watch shrink. The Step 1 correlation results stand as the pre-clean motivating evidence
  only.
- **Over-stripping / content-loss guard.** This is a new check that validates the *cleaning
  function itself*, not the diagnostic's old bugs — worth taking seriously, since a bug here
  would quietly degrade descriptions while looking like an improvement. The specific risk: an
  HTML parser (used in the candidate function, see Step 3) can misinterpret text that merely
  *looks* like a tag — things like `<3`, `x < y`, `^_^<` — and silently delete content that was
  never actually markup. Two complementary checks, both worth doing since they catch different
  failure modes:
  1. For every row, compute `len(cleaned) / len(raw)` (using the null/empty-safe logic from
     the Step 1 fix). Sort ascending and print the ~15–20 rows with the most severe drop that
     *isn't* explained by the raw text being mostly markup — i.e. flag cases where few tags
     were detected but a lot of length still disappeared. Manually eyeball these for silent
     content loss. This catches unanticipated failure patterns actually present in the real
     corpus.
  2. Separately, run a small curated smoke-test list through `clean_description_text()`
     directly — hand-written strings covering known false-positive-prone patterns: e.g.
     `"I loved this book <3"`, `"if x < y > z: ..."`, `"a 5<10 rating scale"`,
     emoji-adjacent angle brackets. Confirm none of them lose meaningful content. This catches
     known risky patterns with certainty, even if they're too rare (or absent) in the real
     corpus to surface via #1.
- No CI wiring — this is a manual run-and-read exercise, consistent with the diagnostic's
  throwaway scope from Step 1.

---

## Step 3 — Implement in the real pipeline

**Final function**, added to `goodreads_ranker/core/utils.py` (the module that already holds
other small text/data-shaping helpers used across the pipeline):

```python
def clean_description_text(text: str | None) -> str:
    """Normalize a raw Goodreads description (HTML-formatted, possibly
    mis-encoded) into plain text suitable for embedding."""
    if not text:
        return ""

    import ftfy
    from bs4 import BeautifulSoup

    text = ftfy.fix_text(text, normalization="NFKC")      # mojibake, control/format chars, NFKC
    text = BeautifulSoup(text, "html.parser").get_text()  # tags + entities
    return re.sub(r"\s+", " ", text).strip()               # whitespace
```

`ftfy.fix_text()` handles mojibake correction, BOM/control-character removal, and Unicode
normalization in one call. `BeautifulSoup(...).get_text()` strips HTML tags *and* unescapes
HTML entities in one call (entity decoding happens automatically during HTML parsing). Order
matters slightly: fixing encoding issues before parsing HTML is safer than the reverse, since a
garbled byte sequence sitting next to a tag delimiter is best resolved first.

- If Step 2's over-stripping guard surfaces a *real, recurring* failure pattern in the actual
  corpus (not just the synthetic smoke-test cases from Step 2), add a minimal, targeted guard
  for that specific pattern only. Don't preemptively generalize beyond what the data actually
  showed — that's the "simple vs. more comprehensive" fork this whole plan was designed to
  resolve with evidence rather than guesswork.
- **Wire into `goodreads_ranker/ml/embedder.py`:** it currently builds the embedding input
  text with `desc_clean = re.sub(r"\s+", " ", desc_raw).strip()`. Replace that line with
  `desc_clean = utils.clean_description_text(r["description"])`, and add `utils` to the
  existing `from goodreads_ranker.core import config, db` import at the top of the file.
- **Dependencies:** add `beautifulsoup4` and `ftfy` to `pyproject.toml`.
- **The raw `description` column in the DB stays untouched.** Cleaning only happens at
  embedding-input time inside `embedder.py`, consistent with the pattern already used there
  (it already does a whitespace-only clean at this same point — this just extends it).
- **Rollout:** no migration code is needed. Changing the cleaning logic changes every row's
  `text_hash`, so the existing `find_stale_or_missing_embeddings()` logic in `embedder.py`
  will automatically mark all ~22k books stale on the next `python main.py embed` run — this
  is the same content-hash mechanism that already exists for regular re-crawls. Budget the
  Ollama time for what amounts to a full re-embed of the corpus.
- After that finishes, `python main.py friend_similarity` and `python main.py predict` pick up
  the new vectors automatically with no code changes — they always read whatever's currently
  in `book_embeddings`.
- **Disposition of the diagnostic script:** once Step 2 passes, delete it or leave it
  unpromoted wherever it currently sits in `old/` — it was explicitly scoped as throwaway from
  the start and isn't referenced by anything else in the codebase.

---

## Explicitly out of scope
- Formal p-value hypothesis testing (superseded by effect-size reporting — see Step 1)
- Fixing the diagnostic's cosmetic bugs (entity regex false positives, URL-in-tag
  overcounting, closing-tag undercounting)
- CI integration / turning the diagnostic into a permanent regression test
- Re-measuring the tag-presence confound post-clean — structurally moot, since cleaning
  removes the tag-presence axis entirely (every row becomes tag-free)
