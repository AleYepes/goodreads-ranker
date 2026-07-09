# Goodreads GraphQL API — Exploration Findings

Endpoint: `https://kxbwmqov6jgg3daaamb744ycu4.appsync-api.us-east-1.amazonaws.com/graphql`
Auth: static `x-api-key` header (AWS AppSync). No JWT required for public book data.

## 1. Schema access

- **Introspection is disabled.** The `__Type` meta-type has no queryable fields (`name`, `kind`, `fields` all error as `FieldUndefined`). No `__schema`/`__type` dump is possible. All field discovery has to be done by trial-and-error probing.
- Unknown-argument and unknown-field errors are informative and consistent (`FieldUndefined`, `UnknownArgument`, `WrongType`, or a general "field not defined for input object type X"), which makes guess-and-check viable despite no introspection.
- Only two entry-point queries confirmed so far: `getBookByLegacyId(legacyId: Int!)` and `getSimilarBooks(id: ID!, pagination: {limit: Int!})`. `getWorkByLegacyId` does **not** exist — Work has no standalone entry point.

## 2. Book vs. Work — critical distinction

- **Book** = a specific edition/printing. Has edition-specific fields: `title`, `titleComplete`, `description`, `primaryContributorEdge`, `secondaryContributorEdges`, `bookSeries`, `bookGenres`, `details` (isbn, format, numPages, publisher, publicationTime, language).
- **Work** = the abstract creative work. Only has: `stats` (aggregate ratings/reviews across all editions), `details` (originalTitle, publicationTime, awardsWon, webUrl, shelvesUrl — *no title/contributor/genre fields exist on Work*), `bestBook`, `editions`.
- **You cannot get Book-level fields (title, author, genre, series, ISBN, format) from Work.** Attempting `work.primaryContributorEdge` etc. fails with `FieldUndefined` — Work genuinely has no such fields, this isn't a bug in our query.
- Practical implication: canonical/display data (title, author, genre) must come from a **Book** fetch — either the originally-seeded book, or (preferably) the `bestBook`.

## 3. ID space mismatch — `work.id`/`work.legacyId` are NOT compatible with Book IDs

- **Confirmed dangerous:** feeding `work.legacyId` (e.g. `41335427`) into `getBookByLegacyId` does **not** error — it silently returns a completely unrelated book ("Praedecessores Nostros"). No validation, no null, just wrong data.
- **Decision:** never use `work.id` / `work.legacyId` to look up a Book. Treat these purely as an internal opaque grouping key if needed at all — do not expose them anywhere a developer might mistake them for a Book reference. Simplest and safest: don't store them at all.
- **`bestBook.id` / `bestBook.legacyId` ARE compatible with Book IDs** — confirmed via a clean `getBookByLegacyId(legacyId: bestBook.legacyId)` fetch that returned full, correct book data (title, contributor, genres, format details) for the canonical Harry Potter 6 edition.

## 4. Canonical book resolution strategy (finalized)

Given the above, the flow for turning an arbitrary seed `legacy_book_id` into canonical data is:

1. `getBookByLegacyId(legacyId: seed_id)` — pulls `work.bestBook.legacyId` alongside normal book fields.
2. If `bestBook.legacyId != seed_id`, issue a second `getBookByLegacyId(legacyId: bestBook.legacyId)` fetch.
3. Store the **bestBook** fetch's Book row as canonical (title, contributors, genres, series, details). The Work row stores only aggregate stats/awards, keyed by `work.id` (the opaque `kca://work/...` string — safe to store as a grouping key, just never dereference it as a book lookup).

This means 1–2 requests to resolve canonical data per work, independent of edition count.

## 5. Editions enumeration (sibling dedup) — mechanics found

Goal: get sibling edition `legacyId`s for a work, to dequeue redundant crawls (e.g. don't separately crawl 673 editions of the same Harry Potter book).

- `work.editions` is a connection: `{ totalCount, edges { node { id legacyId title } }, pageInfo { nextPageToken } }`.
- **`limit` is hard-capped at exactly 20.** Any value ≠ 20 (tested 21 through 500) returns a silent empty/null shape (`totalCount: null, edges: [], nextPageToken: null`) — not an error, just nothing. No way to widen the page size.
- **Pagination input field is `after`** (a `String`), not `page`, `cursor`, `token`, `offset`, `per_page`, etc. — all of those are rejected as unknown fields on `PaginationInput`. Confirmed shape: `pagination: { limit: 20, after: "<token>" }`.
- **The token is not a real opaque cursor — it's forgeable.** `nextPageToken` values are just base64 of `{"next_page": N}` (e.g. `eyJuZXh0X3BhZ2UiOjJ9` → `{"next_page":2}`). We successfully constructed our own tokens for arbitrary page numbers without ever receiving them from a prior response.
- **HARD CAP: only the first 10 pages (200 editions total) are ever retrievable, regardless of method.** Pages 1–10 return data reliably (confirmed both concurrently and sequentially, isolated per-request, spaced 1.5s apart, in randomized order). Pages 11+ always return the same silent empty shape as an out-of-range page, no matter the request pattern — ruled out concurrency artifacts, rate limiting, and connection/session state as causes via isolated single-request tests.
- **A second top-level field, `getEditions(id: ID!, pagination: PaginationInput)`, was found** (accepts the Work's `id`, not a Book id) but it hits the exact same resolver/cap — same default page-1 output, same silent-empty behavior once any explicit `pagination` is passed. Not a workaround, just an alias.
- **The Goodreads website's SSR editions page** (`goodreads.com/work/editions/{workLegacyId}?per_page=100`) is a different, non-GraphQL, non-Next.js code path (no `__NEXT_DATA__` present) that likely queries the legacy DB directly and is not capped at 200. HTML scraping this page was considered as a full-coverage fallback but its own `per_page` parameter is unreliable (requested 100, got 96 and 95 unique ids on two consecutive pages) — an extra layer of undocumented quirks, plus a second scraping surface (different domain, different failure/format-drift modes) to build and maintain.

**Decision: accept partial coverage, capped at 200 editions (10 pages) via GraphQL only. No HTML scraping fallback.** Rationale: the project's actual goal is minimizing redundant crawls/storage, not exhaustive completeness. 673-edition outliers are rare (major bestsellers only); most personal-library books will have single- to low-double-digit edition counts, comfortably under the cap. Any sibling edition beyond the 200-cap that does get crawled "cold" is still recognized as belonging to the same work after the fact (via `work.id` on its own row) — it's a minor crawl inefficiency for outliers, not a correctness or data-quality problem. This keeps the crawler on a single data source/code path.

## 6. Confirmed reusable query fragments

**Editions page fetch (parameterized):**
```graphql
query getBookByLegacyId($legacyBookId: Int!, $pagination: PaginationInput!) {
  getBookByLegacyId(legacyId: $legacyBookId) {
    work {
      editions(pagination: $pagination) {
        totalCount
        edges { node { legacyId title } }
        pageInfo { nextPageToken }
      }
    }
  }
}
```
`pagination` variable: `{"limit": 20, "after": "<base64 of {"next_page": N}>"}` (omit `after` for page 1).

**Loop bound:** fetch pages 1 through `min(ceil(totalCount / 20), 10)`. Never request page 11+; it will silently return nothing and just wastes a request. Pages can be fetched concurrently (forged tokens are valid out of order) since only 10 requests max are ever needed per work.

**Canonical bestBook fetch:**
```graphql
query getBookByLegacyId($legacyBookId: Int!) {
  getBookByLegacyId(legacyId: $legacyBookId) {
    id
    legacyId
    title
    titleComplete
    primaryContributorEdge { node { id legacyId name } role }
    bookGenres { genre { name } }
    details { isbn13 format numPages }
  }
}
```