import argparse
import json
import re
import sqlite3
import unicodedata
import os
import sys
from collections import Counter

# Add parent directory to sys.path so it can find goodreads_ranker when executed from old/ or project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from goodreads_ranker.core.utils import clean_description_text

HTML_TAG_RE = re.compile(r"<\s*([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>")
ENTITY_RE = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{2,}")
MOJIBAKE_MARKERS = ("Ã", "â€", "Â", "ð", "\ufffd")  # \ufffd = U+FFFD replacement char


def check_raw_nul_bytes(path):
    """Warn if the raw CSV file contains NUL bytes: pandas' CSV parser silently
    truncates strings at NUL, which would otherwise hide this issue entirely."""
    if not path.endswith(".csv"):
        return None
    with open(path, "rb") as f:
        raw = f.read()
    return raw.count(b"\x00")


def load_data(path, table=None, column="description", query=None):
    if path.endswith((".db", ".sqlite", ".sqlite3")):
        conn = sqlite3.connect(path)
        try:
            if query:
                df = pd.read_sql_query(query, conn)
            elif table == "books" and column == "description":
                # Dedicated query joining books with genres, publisher, language, etc.
                dedicated_query = """
                SELECT b.legacy_id, b.description, b.publisher, b.language_name, b.publication_time,
                       b.original_publication_time,
                       b.star_1, b.star_2, b.star_3, b.star_4, b.star_5,
                       GROUP_CONCAT(g.name, '|') AS genres_list
                FROM books b
                LEFT JOIN book_genres bg ON b.legacy_id = bg.book_id
                LEFT JOIN genres g ON bg.genre_id = g.legacy_id
                GROUP BY b.legacy_id
                """
                df = pd.read_sql_query(dedicated_query, conn)
            else:
                if not table:
                    raise ValueError("Must supply --table for sqlite input unless --query is given")
                df = pd.read_sql_query(f"SELECT {column} FROM {table}", conn)
        finally:
            conn.close()
    else:
        df = pd.read_csv(path)

    # Compute rating count and average rating if star columns are present
    star_cols = ["star_1", "star_2", "star_3", "star_4", "star_5"]
    if all(col in df.columns for col in star_cols):
        df["rating_count"] = df[star_cols].sum(axis=1)
        total_stars = df["star_1"] * 1 + df["star_2"] * 2 + df["star_3"] * 3 + df["star_4"] * 4 + df["star_5"] * 5
        df["avg_rating"] = np.where(df["rating_count"] > 0, total_stars / df["rating_count"], np.nan)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
    return df


def analyze_text(text):
    """Return a dict of findings for a single description string."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return {"is_null": True, "is_empty": False}
    if isinstance(text, str) and text == "":
        return {"is_null": False, "is_empty": True}

    findings = {"is_null": False, "is_empty": False, "length": len(text)}

    findings["html_tags"] = HTML_TAG_RE.findall(text)
    findings["html_entities"] = ENTITY_RE.findall(text)

    control_chars, format_chars, other_chars = [], [], []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cc" and ch not in ("\n", "\t"):
            control_chars.append(ch)
        elif cat == "Cf":
            format_chars.append(ch)
        elif cat.startswith("C"):
            other_chars.append(ch)
    findings["control_chars"] = control_chars
    findings["format_chars"] = format_chars
    findings["other_control_category_chars"] = other_chars

    findings["has_multi_space"] = bool(MULTI_SPACE_RE.search(text))
    findings["has_multi_newline"] = bool(MULTI_NEWLINE_RE.search(text))
    findings["leading_trailing_ws"] = text != text.strip()
    findings["urls"] = URL_RE.findall(text)

    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    findings["non_ascii_ratio"] = non_ascii / len(text) if text else 0
    findings["possible_mojibake"] = any(m in text for m in MOJIBAKE_MARKERS)
    findings["nfkc_changes_text"] = unicodedata.normalize("NFKC", text) != text

    return findings


def aggregate(df, column, sample_n=3):
    total = len(df)
    stats = Counter()
    tag_counter, entity_counter = Counter(), Counter()
    control_char_counter, format_char_counter = Counter(), Counter()
    lengths = []
    examples = {k: [] for k in ["html_tags", "control_chars", "format_chars", "mojibake", "multi_space", "urls"]}

    for val in df[column]:
        f = analyze_text(val)
        if f.get("is_null") or f.get("is_empty"):
            if f.get("is_null"):
                stats["null"] += 1
            if f.get("is_empty"):
                stats["empty"] += 1
            stats["null_or_empty"] += 1
            continue

        lengths.append(f["length"])

        if f["html_tags"]:
            stats["contains_html_tags"] += 1
            tag_counter.update(t.lower() for t in f["html_tags"])
            if len(examples["html_tags"]) < sample_n:
                examples["html_tags"].append(val[:200])
        if f["html_entities"]:
            stats["contains_html_entities"] += 1
            entity_counter.update(f["html_entities"])
        if f["control_chars"]:
            stats["contains_control_chars"] += 1
            control_char_counter.update(f["control_chars"])
            if len(examples["control_chars"]) < sample_n:
                examples["control_chars"].append(repr(val[:200]))
        if f["format_chars"]:
            stats["contains_format_chars"] += 1
            format_char_counter.update(f["format_chars"])
            if len(examples["format_chars"]) < sample_n:
                examples["format_chars"].append(repr(val[:200]))
        if f["other_control_category_chars"]:
            stats["contains_other_control_category"] += 1
        if f["urls"]:
            stats["contains_urls"] += 1
            if len(examples["urls"]) < sample_n:
                examples["urls"].append(val[:200])
        if f["has_multi_space"]:
            stats["has_multi_space"] += 1
            if len(examples["multi_space"]) < sample_n:
                examples["multi_space"].append(repr(val[:200]))
        if f["has_multi_newline"]:
            stats["has_multi_newline"] += 1
        if f["leading_trailing_ws"]:
            stats["leading_trailing_ws"] += 1
        if f["possible_mojibake"]:
            stats["possible_mojibake"] += 1
            if len(examples["mojibake"]) < sample_n:
                examples["mojibake"].append(val[:200])
        if f["nfkc_changes_text"]:
            stats["nfkc_changes_text"] += 1

    length_stats = {}
    if lengths:
        ls = sorted(lengths)
        n = len(ls)
        length_stats = {"min": ls[0], "max": ls[-1], "mean": sum(ls) / n, "median": ls[n // 2]}

    return {
        "total_rows": total,
        "stats": dict(stats),
        "length_stats": length_stats,
        "top_tags": tag_counter.most_common(20),
        "top_entities": entity_counter.most_common(20),
        "top_control_chars": [(repr(c), n) for c, n in control_char_counter.most_common(20)],
        "top_format_chars": [(repr(c), n) for c, n in format_char_counter.most_common(20)],
        "examples": examples,
    }


def print_report(result):
    total = result["total_rows"]
    stats = result["stats"]

    def pct(key):
        n = stats.get(key, 0)
        return f"{n} ({n / total * 100:.1f}%)" if total else str(n)

    print(f"Total rows: {total}")
    print(f"Null/empty: {pct('null_or_empty')} (Null: {stats.get('null', 0)}, Empty: {stats.get('empty', 0)})\n")

    ls = result["length_stats"]
    print("-- Length statistics (non-null, characters) --")
    if ls:
        print(f"  min={ls['min']}  median={ls['median']}  mean={ls['mean']:.1f}  max={ls['max']}")
    print()

    print("-- Structural issues found --")
    print(f"  Contains HTML tags:                {pct('contains_html_tags')}")
    print(f"  Contains HTML entities:            {pct('contains_html_entities')}")
    print(f"  Contains control chars (Cc):       {pct('contains_control_chars')}")
    print(f"  Contains format chars (Cf):        {pct('contains_format_chars')}")
    print(f"  Other control-category chars:      {pct('contains_other_control_category')}")
    print(f"  Contains URLs:                     {pct('contains_urls')}")
    print(f"  Repeated spaces/tabs:              {pct('has_multi_space')}")
    print(f"  Repeated blank lines:              {pct('has_multi_newline')}")
    print(f"  Leading/trailing whitespace:       {pct('leading_trailing_ws')}")
    print(f"  Possible mojibake/encoding issues: {pct('possible_mojibake')}")
    print(f"  NFKC normalization would change text: {pct('nfkc_changes_text')}")
    print()

    if result["top_tags"]:
        print("-- Most common HTML tags --")
        for tag, n in result["top_tags"]:
            print(f"  <{tag}>: {n}")
        print()

    if result["top_entities"]:
        print("-- Most common HTML entities --")
        for ent, n in result["top_entities"]:
            print(f"  &{ent};: {n}")
        print()

    if result["top_control_chars"]:
        print("-- Most common control characters (Cc) --")
        for c, n in result["top_control_chars"]:
            print(f"  {c}: {n}")
        print()

    if result["top_format_chars"]:
        print("-- Most common format characters (Cf: zero-width/bidi/etc.) --")
        for c, n in result["top_format_chars"]:
            print(f"  {c}: {n}")
        print()

    print("-- Sample offending rows --")
    for key, samples in result["examples"].items():
        if samples:
            print(f"  [{key}]")
            for s in samples:
                print(f"    {s}")
    print()


def print_recommendations(stats):
    print("=" * 70)
    print("RECOMMENDED PREPROCESSING STEPS")
    print("=" * 70)
    recs = []
    if stats.get("contains_html_tags"):
        recs.append(
            "Strip HTML tags, e.g. BeautifulSoup(text, 'html.parser').get_text() "
            "(preferred over regex — handles malformed markup safely)."
        )
    if stats.get("contains_html_entities"):
        recs.append("Unescape HTML entities with html.unescape() before/alongside tag stripping.")
    if stats.get("contains_control_chars") or stats.get("contains_other_control_category"):
        recs.append(
            "Remove non-printable control characters, e.g. "
            "''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\\n\\t')."
        )
    if stats.get("contains_format_chars"):
        recs.append(
            "Strip Unicode format characters (category 'Cf': zero-width spaces, joiners, "
            "bidi marks) — they add no semantic value and can confuse tokenizers."
        )
    if stats.get("nfkc_changes_text"):
        recs.append(
            "Apply unicodedata.normalize('NFKC', text) to fold compatibility characters "
            "(fullwidth forms, ligatures, etc.) into standard forms."
        )
    if stats.get("possible_mojibake"):
        recs.append(
            "Investigate mojibake/double-encoding (e.g. UTF-8 bytes decoded as Latin-1). "
            "The 'ftfy' library's fix_text() is a good targeted fix; also check ingestion encoding."
        )
    if stats.get("has_multi_space") or stats.get("has_multi_newline") or stats.get("leading_trailing_ws"):
        recs.append("Normalize whitespace: collapse repeats and strip ends, e.g. re.sub(r'\\s+', ' ', text).strip().")
    if stats.get("contains_urls"):
        recs.append(
            "Decide whether to keep, mask, or drop URLs depending on whether they carry "
            "semantic signal for your embedding use case."
        )
    if not recs:
        recs.append("No major issues detected; light whitespace normalization is still worth doing.")

    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r}")

    print("\nSuggested cleaning pipeline (in order):")
    print("  1. html.unescape(text)")
    print("  2. Strip HTML tags (BeautifulSoup or regex)")
    print("  3. unicodedata.normalize('NFKC', text)")
    print("  4. Remove Cc/Cf characters except \\n and \\t")
    print("  5. Collapse whitespace and .strip()")
    print("  6. Optionally handle/remove URLs")


def run_correlation_check(df, column="description"):
    """
    Perform correlation checks between description tag-presence and other metadata.
    """
    df_valid = df[df[column].apply(lambda x: isinstance(x, str) and x != "")].copy()
    if len(df_valid) == 0:
        print("No valid descriptions to run correlation check.")
        return

    df_valid["has_html_tags"] = df_valid[column].apply(lambda x: bool(HTML_TAG_RE.search(x)))

    df_tagged = df_valid[df_valid["has_html_tags"]]
    df_untagged = df_valid[~df_valid["has_html_tags"]]

    n_tagged = len(df_tagged)
    n_untagged = len(df_untagged)

    print("=" * 70)
    print("HTML-TAG-PRESENCE CORRELATION CHECK (PRE-CLEAN)")
    print("=" * 70)
    print(f"Tagged group size:   {n_tagged} ({n_tagged / len(df_valid) * 100:.1f}%)")
    print(f"Untagged group size: {n_untagged} ({n_untagged / len(df_valid) * 100:.1f}%)")
    print()

    if n_tagged == 0 or n_untagged == 0:
        print("One of the groups is empty, skipping correlation analysis.")
        return

    pub_col = "publication_time"
    if "publication_time" in df_valid.columns and "original_publication_time" in df_valid.columns:
        pub_nulls = df_valid["publication_time"].isna().sum()
        orig_pub_nulls = df_valid["original_publication_time"].isna().sum()
        if orig_pub_nulls < pub_nulls:
            pub_col = "original_publication_time"

    # A. Continuous Variables
    continuous_cols = []
    for c in ["rating_count", "avg_rating", pub_col]:
        if c in df_valid.columns:
            continuous_cols.append(c)

    if continuous_cols:
        print("-- Continuous Variables: Median, IQR, and Rank-Biserial Correlation --")
        for col in continuous_cols:
            y_tagged = df_tagged[col].dropna()
            y_untagged = df_untagged[col].dropna()

            med_t = y_tagged.median()
            q75_t, q25_t = y_tagged.quantile(0.75), y_tagged.quantile(0.25)
            iqr_t = q75_t - q25_t

            med_u = y_untagged.median()
            q75_u, q25_u = y_untagged.quantile(0.75), y_untagged.quantile(0.25)
            iqr_u = q75_u - q25_u

            n_t = len(y_tagged)
            n_u = len(y_untagged)
            if n_t > 0 and n_u > 0:
                u_stat, _ = mannwhitneyu(y_tagged, y_untagged, alternative="two-sided")
                r_rb = 1.0 - (2.0 * u_stat) / (n_t * n_u)
            else:
                r_rb = float("nan")

            print(f"  {col}:")
            print(f"    Tagged group (N={n_t}):   median={med_t:.2f}, IQR={iqr_t:.2f}")
            print(f"    Untagged group (N={n_u}): median={med_u:.2f}, IQR={iqr_u:.2f}")
            print(f"    Rank-Biserial Correlation: {r_rb:.4f}")
            print()

    # B. Genre (multi-label)
    if "genres_list" in df_valid.columns:

        def parse_genres(val):
            if not val or pd.isna(val):
                return []
            return [g.strip() for g in val.split("|") if g.strip()]

        df_valid["genres"] = df_valid["genres_list"].apply(parse_genres)

        all_genres = [g for genres in df_valid["genres"] for g in genres]
        genre_counts = Counter(all_genres)
        frequent_genres = [g for g, count in genre_counts.items() if count >= 30]

        genre_stats = []
        df_tagged_genres = df_valid[df_valid["has_html_tags"]]
        df_untagged_genres = df_valid[~df_valid["has_html_tags"]]

        for genre in frequent_genres:
            count_tagged = sum(1 for genres in df_tagged_genres["genres"] if genre in genres)
            count_untagged = sum(1 for genres in df_untagged_genres["genres"] if genre in genres)
            p_tagged = count_tagged / n_tagged
            p_untagged = count_untagged / n_untagged
            gap = abs(p_tagged - p_untagged) * 100
            genre_stats.append({"genre": genre, "p_tagged": p_tagged * 100, "p_untagged": p_untagged * 100, "gap": gap})

        genre_stats = sorted(genre_stats, key=lambda x: x["gap"], reverse=True)

        print("-- Genre (Top 20 absolute percentage-point gaps) --")
        print(f"  {'Genre':<30} | {'% in Tagged':<12} | {'% in Untagged':<14} | {'Gap (pp)':<8}")
        print("  " + "-" * 75)
        for stat in genre_stats[:20]:
            print(
                f"  {stat['genre']:<30} | {stat['p_tagged']:>10.1f}% | {stat['p_untagged']:>12.1f}% | {stat['gap']:>7.1f}"
            )
        print()

    # C. Publisher and language_name
    for col in ["publisher", "language_name"]:
        if col in df_valid.columns:
            series_filled = df_valid[col].fillna("Unknown")
            top_vals = series_filled.value_counts().index[:20]
            df_valid[col + "_bucket"] = series_filled.apply(lambda x: x if x in top_vals else "Other")

            df_tagged_cat = df_valid[df_valid["has_html_tags"]]
            df_untagged_cat = df_valid[~df_valid["has_html_tags"]]

            cat_stats = []
            buckets = df_valid[col + "_bucket"].unique()
            for b in buckets:
                c_tagged = sum(df_tagged_cat[col + "_bucket"] == b)
                c_untagged = sum(df_untagged_cat[col + "_bucket"] == b)
                p_tagged = c_tagged / n_tagged
                p_untagged = c_untagged / n_untagged
                gap = abs(p_tagged - p_untagged) * 100
                cat_stats.append({"value": b, "p_tagged": p_tagged * 100, "p_untagged": p_untagged * 100, "gap": gap})
            cat_stats = sorted(cat_stats, key=lambda x: x["gap"], reverse=True)

            print(f"-- {col.capitalize()} (buckets sorted by absolute percentage-point gaps) --")
            print(f"  {col.capitalize():<30} | {'% in Tagged':<12} | {'% in Untagged':<14} | {'Gap (pp)':<8}")
            print("  " + "-" * 75)
            for stat in cat_stats:
                print(
                    f"  {stat['value']:<30} | {stat['p_tagged']:>10.1f}% | {stat['p_untagged']:>12.1f}% | {stat['gap']:>7.1f}"
                )
            print()


def run_content_loss_guards(df_raw, df_clean, column):
    print("=" * 70)
    print("CONTENT-LOSS / OVER-STRIPPING GUARDS")
    print("=" * 70)

    # 1. Curated False-Positive Smoke Tests
    print("-- 1. Curated False-Positive Smoke Tests --")
    smoke_tests = [
        "I loved this book <3",
        "if x < y > z: ...",
        "a 5<10 rating scale",
        "this is a test with emoji-adjacent angle brackets: ^_^<3",
        "HTML tags <br /> should be stripped but text kept.",
        "Entities &amp; and &lt; should be decoded to & and <.",
    ]

    for t in smoke_tests:
        cleaned = clean_description_text(t)
        print(f"  Raw:     {t}")
        print(f"  Cleaned: {cleaned}")
        print()

    # 2. Real Corpus Over-Stripping Check
    print("-- 2. Real Corpus Over-Stripping Check --")
    print("     (Flagging rows with len(raw) >= 50, tag_count <= 1, sorted by largest length reduction)")

    records = []
    for i in range(len(df_raw)):
        raw_val = df_raw.iloc[i][column]
        clean_val = df_clean.iloc[i][column]

        if raw_val is None or pd.isna(raw_val) or not isinstance(raw_val, str) or raw_val == "":
            continue

        len_raw = len(raw_val)
        if len_raw < 50:
            continue

        len_clean = len(clean_val)
        ratio = len_clean / len_raw

        # Count HTML tags
        tag_count = len(HTML_TAG_RE.findall(raw_val))

        if tag_count <= 1:
            records.append(
                {
                    "raw": raw_val,
                    "cleaned": clean_val,
                    "len_raw": len_raw,
                    "len_clean": len_clean,
                    "ratio": ratio,
                    "tag_count": tag_count,
                }
            )

    if not records:
        print("  No rows matched the criteria for checking over-stripping (len(raw) >= 50, tag_count <= 1).")
    else:
        # Sort by ratio ascending
        records = sorted(records, key=lambda x: x["ratio"])
        top_n = min(20, len(records))
        print(f"  Top {top_n} rows with highest unexplained length drop:")
        print(f"  {'Ratio':<5} | {'Raw Len':<7} | {'Clean Len':<9} | {'Snippet'}")
        print("  " + "-" * 75)
        for r in records[:top_n]:
            snippet = repr(r["raw"][:80])
            print(f"  {r['ratio']:>5.2f} | {r['len_raw']:>7} | {r['len_clean']:>9} | {snippet}")
            clean_snippet = repr(r["cleaned"][:80])
            print(f"        | {' ':<7} | {' ':<9} | -> {clean_snippet}")
            print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "path",
        nargs="?",
        default="data/goodreads.db",
        help="Path to CSV file or SQLite .db/.sqlite file (default: data/goodreads.db)",
    )
    parser.add_argument("--table", default="books", help="Table name (for SQLite input, default: books)")
    parser.add_argument("--column", default="description", help="Column to analyze (default: description)")
    parser.add_argument("--query", help="Custom SQL query for SQLite input (overrides --table)")
    parser.add_argument("--sample", type=int, default=3, help="Example rows to show per issue type")
    parser.add_argument("--json-out", help="Optional path to write results as JSON")
    args = parser.parse_args()

    nul_count = check_raw_nul_bytes(args.path)
    if nul_count:
        print(
            f"WARNING: raw file contains {nul_count} NUL byte(s). pandas' CSV parser silently "
            f"truncates strings at NUL, so affected rows may show as shorter/cleaner than they "
            f"really are. Investigate the source export/encoding before trusting length stats "
            f"below.\n"
        )

    df = load_data(args.path, table=args.table, column=args.column, query=args.query)

    # Run the correlation check first on the raw DataFrame
    run_correlation_check(df, args.column)

    # Run BEFORE report
    print("=" * 70)
    print("BEFORE CLEANING REPORT")
    print("=" * 70)
    raw_result = aggregate(df, args.column, sample_n=args.sample)
    print_report(raw_result)

    # Generate cleaned DataFrame in-memory
    df_clean = df.copy()
    df_clean[args.column] = df_clean[args.column].apply(
        lambda x: clean_description_text(x) if isinstance(x, str) else x
    )

    # Run AFTER report
    print("=" * 70)
    print("AFTER CLEANING REPORT")
    print("=" * 70)
    clean_result = aggregate(df_clean, args.column, sample_n=args.sample)
    print_report(clean_result)

    # Run content loss guards
    run_content_loss_guards(df, df_clean, args.column)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(clean_result, f, indent=2, default=str)
        print(f"\nFull results written to {args.json_out}")


if __name__ == "__main__":
    main()
