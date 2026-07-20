import os
import re
import sqlite3
import sys
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from goodreads_ranker.core.utils import clean_description_text

HTML_TAG_RE = re.compile(r"<\s*([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>")
ENTITY_RE = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{2,}")
MOJIBAKE_MARKERS = ("Ã", "â€", "Â", "ð", "\ufffd")
DB_PATH = "data/goodreads.db"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT b.legacy_id, b.description, b.publisher, b.language_name, b.publication_time,
           b.original_publication_time,
           b.star_1, b.star_2, b.star_3, b.star_4, b.star_5,
           GROUP_CONCAT(g.name, '|') AS genres_list
    FROM books b
    LEFT JOIN book_genres bg ON b.legacy_id = bg.book_id
    LEFT JOIN genres g ON bg.genre_id = g.legacy_id
    GROUP BY b.legacy_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    star_cols = ["star_1", "star_2", "star_3", "star_4", "star_5"]
    if all(col in df.columns for col in star_cols):
        df["rating_count"] = df[star_cols].sum(axis=1)
        total_stars = df["star_1"] * 1 + df["star_2"] * 2 + df["star_3"] * 3 + df["star_4"] * 4 + df["star_5"] * 5
        df["avg_rating"] = np.where(df["rating_count"] > 0, total_stars / df["rating_count"], np.nan)
    return df


def analyze_text(text):
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


def aggregate(df, sample_n=3):
    total = len(df)
    stats = Counter()
    tag_counter, entity_counter = Counter(), Counter()
    control_char_counter, format_char_counter = Counter(), Counter()
    lengths = []
    examples = {k: [] for k in ["html_tags", "control_chars", "format_chars", "mojibake", "multi_space", "urls"]}

    for val in df["description"]:
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


def print_report(result, after=False):
    total = result["total_rows"]
    stats = result["stats"]

    def pct(key):
        n = stats.get(key, 0)
        return f"{n} ({n / total * 100:.1f}%)" if total else str(n)

    print(f"Total rows: {total}")
    print(f"Null/empty: {pct('null_or_empty')} (Null: {stats.get('null', 0)}, Empty: {stats.get('empty', 0)})\n")

    ls = result["length_stats"]
    if ls:
        print(f"Length statistics: min={ls['min']} median={ls['median']} mean={ls['mean']:.1f} max={ls['max']}\n")

    print(f"Contains HTML tags: {pct('contains_html_tags')}")
    print(f"Contains HTML entities: {pct('contains_html_entities')}")
    print(f"Contains control chars (Cc): {pct('contains_control_chars')}")
    print(f"Contains format chars (Cf): {pct('contains_format_chars')}")
    print(f"Other control-category chars: {pct('contains_other_control_category')}")
    print(f"Contains URLs: {pct('contains_urls')}")
    print(f"Repeated spaces/tabs: {pct('has_multi_space')}")
    print(f"Repeated blank lines: {pct('has_multi_newline')}")
    print(f"Leading/trailing whitespace: {pct('leading_trailing_ws')}")
    print(f"Possible mojibake: {pct('possible_mojibake')}")
    print(f"NFKC normalization changes: {pct('nfkc_changes_text')}\n")

    if result["top_tags"]:
        print("Top HTML tags:", ", ".join(f"<{t}>: {n}" for t, n in result["top_tags"]))
    if result["top_entities"]:
        print("Top HTML entities:", ", ".join(f"&{e};: {n}" for e, n in result["top_entities"]))
    if result["top_control_chars"]:
        print("Top control chars (Cc):", ", ".join(f"{c}: {n}" for c, n in result["top_control_chars"]))
    if result["top_format_chars"]:
        print("Top format chars (Cf):", ", ".join(f"{c}: {n}" for c, n in result["top_format_chars"]))
    print()

    if not after:
        for key, samples in result["examples"].items():
            if samples:
                print(f"Samples [{key}]:")
                for s in samples:
                    print(f"  {s}")
                print()


def run_correlation_check(df):
    df_valid = df[df["description"].apply(lambda x: isinstance(x, str) and x != "")].copy()
    if len(df_valid) == 0:
        print("No valid descriptions to run correlation check.")
        return

    df_valid["has_html_tags"] = df_valid["description"].apply(lambda x: bool(HTML_TAG_RE.search(x)))

    if "genres_list" in df_valid.columns:

        def parse_genres(val):
            return [g.strip() for g in val.split("|") if g.strip()] if val and not pd.isna(val) else []

        df_valid["genres"] = df_valid["genres_list"].apply(parse_genres)

    for col in ["publisher", "language_name"]:
        if col in df_valid.columns:
            series_filled = df_valid[col].fillna("Unknown")
            top_vals = series_filled.value_counts().index[:20]
            df_valid[col + "_bucket"] = series_filled.apply(lambda x: x if x in top_vals else "Other")

    df_tagged = df_valid[df_valid["has_html_tags"]]
    df_untagged = df_valid[~df_valid["has_html_tags"]]
    n_tagged = len(df_tagged)
    n_untagged = len(df_untagged)

    print("HTML-TAG-PRESENCE CORRELATION CHECK (PRE-CLEAN)")
    print(
        f"Tagged: {n_tagged} ({n_tagged / len(df_valid) * 100:.1f}%) | Untagged: {n_untagged} ({n_untagged / len(df_valid) * 100:.1f}%)"
    )

    if n_tagged == 0 or n_untagged == 0:
        return

    pub_col = "publication_time"
    if "publication_time" in df_valid.columns and "original_publication_time" in df_valid.columns:
        if df_valid["original_publication_time"].isna().sum() < df_valid["publication_time"].isna().sum():
            pub_col = "original_publication_time"

    continuous_cols = [c for c in ["rating_count", "avg_rating", pub_col] if c in df_valid.columns]
    for col in continuous_cols:
        y_tagged = df_tagged[col].dropna()
        y_untagged = df_untagged[col].dropna()
        n_t, n_u = len(y_tagged), len(y_untagged)
        if n_t > 0 and n_u > 0:
            u_stat, _ = mannwhitneyu(y_tagged, y_untagged, alternative="two-sided")
            r_rb = 1.0 - (2.0 * u_stat) / (n_t * n_u)
        else:
            r_rb = float("nan")
        print(
            f"  {col}: Tagged Median={y_tagged.median():.2f} (N={n_t}) | Untagged Median={y_untagged.median():.2f} (N={n_u}) | Rank-Biserial Corr={r_rb:.4f}"
        )

    if "genres_list" in df_valid.columns:
        all_genres = [g for genres in df_valid["genres"] for g in genres]
        genre_counts = Counter(all_genres)
        frequent_genres = [g for g, count in genre_counts.items() if count >= 30]

        genre_stats = []
        for genre in frequent_genres:
            p_tagged = sum(1 for genres in df_tagged["genres"] if genre in genres) / n_tagged
            p_untagged = sum(1 for genres in df_untagged["genres"] if genre in genres) / n_untagged
            genre_stats.append({"genre": genre, "gap": abs(p_tagged - p_untagged) * 100})
        genre_stats = sorted(genre_stats, key=lambda x: x["gap"], reverse=True)[:20]
        print("  Top Genre Gaps (pp):", ", ".join(f"{g['genre']} ({g['gap']:.1f}%)" for g in genre_stats))

    for col in ["publisher", "language_name"]:
        if col in df_valid.columns:
            cat_stats = []
            for b in df_valid[col + "_bucket"].unique():
                p_tagged = sum(df_tagged[col + "_bucket"] == b) / n_tagged
                p_untagged = sum(df_untagged[col + "_bucket"] == b) / n_untagged
                cat_stats.append({"value": b, "gap": abs(p_tagged - p_untagged) * 100})
            cat_stats = sorted(cat_stats, key=lambda x: x["gap"], reverse=True)
            print(
                f"  Top {col.capitalize()} Gaps (pp):",
                ", ".join(f"{c['value']} ({c['gap']:.1f}%)" for c in cat_stats[:10]),
            )
    print()


def run_content_loss_guards(df_raw, df_clean):
    print("CONTENT-LOSS / OVER-STRIPPING GUARDS")
    print("Smoke tests:")
    smoke_tests = [
        # --- HTML Tags (Valid, Malformed, and Self-Closing) ---
        "HTML tags <br /> should be stripped but text kept.",
        "<p>This is inside a paragraph tag.</p>",
        '<div class="content-class" style="margin-top: 10px;">Tagged text with attributes</div>',
        "nested tags: <div><span>nested <b>bold</b> text</span></div>",
        "<a href='http://example.com'>Link text</a>",
        "Malformed tags: <b This tag is not closed properly but should not crash.",
        "Case sensitivity in tags: <DIV>All Caps Tag</DIV>",
        "Empty tags: <p></p>text<br>",
        # --- HTML Entities (Named, Decimal, Hex, and Malformed) ---
        "Entities &amp; and &lt; should be decoded to & and <.",
        "Quotes: &quot;double&quot; and &apos;single&apos;.",
        "Curly quotes and dashes: &ldquo;hello&rdquo; and &mdash; dash.",
        "Non-breaking space: text&nbsp;text.",
        "Decimal numeric entity: &#38; should become &.",
        "Hexadecimal numeric entity: &#x26; should become &.",
        "Malformed or incomplete entities: &amp;amp; should decode once, and &amp should be handled.",
        "Non-entities with ampersands: Rock & Roll or AT&T or search?q=a&b=c.",
        # --- Angle Brackets that are NOT HTML (Emoticons, Math, Code, Arrows) ---
        "I loved this book <3",
        "this is a test with emoji-adjacent angle brackets: ^_^<3",
        "Other emoticons: >_< and o_O and <_< and >.>",
        "if x < y > z: ...",
        "a 5<10 rating scale",
        "Mathematical expressions: 3 < 5 and 10 > 2.",
        "Double angle brackets: <<The Title>> or << Previous Page >>",
        "Arrows: <-- back and forward -->",
        # --- Whitespace Normalization (Spaces, Tabs, Newlines, Non-breaking, Zero-width) ---
        "   Leading and trailing space removal.   ",
        "Repeated    spaces     and\ttabs\t\tshould be collapsed.",
        "Repeated\n\n\nnewlines\r\n\r\nand linebreaks.",
        "Mixed spaces: word\u00a0word (non-breaking space) and word\u2002word (en space).",
        "Zero-width space: word\u200bword.",
        # --- Unicode Normalization & Invisible/Control Characters ---
        "Decomposed character: cafe\u0301 (should normalize to café).",
        "Ligatures: \ufb01le (should normalize or remain stable depending on configuration).",
        "Control characters (Cc/Cf) to remove: \x00null, \x07bell, and \x1f.",
        "Directional marks: \u200eright-to-left text boundaries.",
        # --- Mojibake Signatures (Encoding Issues) ---
        "UTF-8 interpreted as Windows-1252: Ã© (é), â€™ ('), Â° (°), ðŸ˜Š (smiley), \ufffd (replacement char).",
        # --- URLs, Domains, and Emails ---
        "Standard secure URL: https://www.example.com/path?query=1",
        "Standard insecure URL: http://example.org",
        "No-protocol URL: www.goodreads.com",
        "Email address: contact@authorwebsite.com",
        "Domain-like text that isn't a URL: MySite.com or test.net.",
        # --- Markdown & Escapes ---
        "Markdown syntax: **bold**, *italic*, _under_, [link](url), # header.",
        "Backslashes and escapes: text\\with\\backslashes.",
        # --- Extreme Cases & Boundaries ---
        "",  # Empty string
        " ",  # Single space
        "&" * 10,  # Repeated entities/ampersands
        "<" * 10,  # Repeated angle brackets
    ]
    for t in smoke_tests:
        print(f"  Raw: {t} -> Clean: {clean_description_text(t)}")

    print("\nReal Corpus Over-Stripping Check (len >= 50, tag_count <= 1):")
    records = []
    for i in range(len(df_raw)):
        raw_val = df_raw.iloc[i]["description"]
        clean_val = df_clean.iloc[i]["description"]

        if not isinstance(raw_val, str) or raw_val == "":
            continue

        len_raw = len(raw_val)
        if len_raw < 50:
            continue

        len_clean = len(clean_val) if isinstance(clean_val, str) else 0
        ratio = len_clean / len_raw
        tag_count = len(HTML_TAG_RE.findall(raw_val))

        if tag_count <= 1:
            records.append(
                {"raw": raw_val, "cleaned": clean_val, "len_raw": len_raw, "len_clean": len_clean, "ratio": ratio}
            )

    if not records:
        print("  No over-stripping candidates found.")
    else:
        records = sorted(records, key=lambda x: x["ratio"])[:20]
        for r in records:
            print(f"  Ratio: {r['ratio']:.2f} | Raw: {r['len_raw']} | Clean: {r['len_clean']}")
            print(f"    Raw:   {repr(r['raw'][:200])}")
            print(f"    Clean: {repr(r['cleaned'][:200])}")
    print()


def print_recommendations(stats):
    print("RECOMMENDED PREPROCESSING STEPS")
    recs = []
    if stats.get("contains_html_tags"):
        recs.append("Strip HTML tags (BeautifulSoup preferred).")
    if stats.get("contains_html_entities"):
        recs.append("Unescape HTML entities with html.unescape().")
    if stats.get("contains_control_chars") or stats.get("contains_other_control_category"):
        recs.append("Remove non-printable control characters.")
    if stats.get("contains_format_chars"):
        recs.append("Strip Unicode format characters (Cf).")
    if stats.get("nfkc_changes_text"):
        recs.append("Apply unicodedata.normalize('NFKC').")
    if stats.get("possible_mojibake"):
        recs.append("Investigate/fix mojibake (e.g. ftfy.fix_text()).")
    if stats.get("has_multi_space") or stats.get("has_multi_newline") or stats.get("leading_trailing_ws"):
        recs.append("Normalize whitespace (strip and collapse).")
    if stats.get("contains_urls"):
        recs.append("Decide whether to keep, mask, or drop URLs.")

    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r}")
    print()


def main():
    df = load_data()

    run_correlation_check(df)

    print("BEFORE CLEANING REPORT")
    raw_result = aggregate(df)
    print_report(raw_result)

    df_clean = df.copy()
    df_clean["description"] = df_clean["description"].apply(
        lambda x: clean_description_text(x) if isinstance(x, str) else x
    )

    print("AFTER CLEANING REPORT")
    clean_result = aggregate(df_clean)
    print_report(clean_result, after=True)

    run_content_loss_guards(df, df_clean)
    print_recommendations(raw_result["stats"])


if __name__ == "__main__":
    main()
