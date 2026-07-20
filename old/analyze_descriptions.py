import re
import sqlite3
import unicodedata
from collections import Counter

DB_PATH = "data/goodreads.db"


def summarize(desc: str):
    if desc is None:
        return {
            "empty": True,
            "len": 0,
            "bytes": 0,
            "html": False,
            "entities": False,
            "controls": 0,
            "format": 0,
            "non_ascii": 0,
            "scripts": Counter(),
        }

    scripts = Counter()

    controls = 0
    formats = 0
    non_ascii = 0

    for ch in desc:
        if ord(ch) > 127:
            non_ascii += 1

        cat = unicodedata.category(ch)
        if cat == "Cc":
            controls += 1
        elif cat == "Cf":
            formats += 1

        name = unicodedata.name(ch, "")
        if name:
            script = name.split()[0]
            scripts[script] += 1

    return {
        "empty": desc == "",
        "len": len(desc),
        "bytes": len(desc.encode("utf-8")),
        "html": bool(re.search(r"<[^>]+>", desc)),
        "entities": bool(re.search(r"&[A-Za-z#0-9]+;", desc)),
        "controls": controls,
        "format": formats,
        "non_ascii": non_ascii,
        "scripts": scripts,
    }


def main():
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute(
        """
        SELECT legacy_id, title, description
        FROM books
        """
    ).fetchall()

    conn.close()

    total = len(rows)

    html_books = []
    entity_books = []
    control_books = []
    format_books = []

    lengths = []
    byte_lengths = []

    script_counter = Counter()

    for legacy_id, title, desc in rows:
        s = summarize(desc)

        lengths.append(s["len"])
        byte_lengths.append(s["bytes"])

        script_counter.update(s["scripts"])

        if s["html"]:
            html_books.append((legacy_id, title))

        if s["entities"]:
            entity_books.append((legacy_id, title))

        if s["controls"]:
            control_books.append((legacy_id, title, s["controls"]))

        if s["format"]:
            format_books.append((legacy_id, title, s["format"]))

    print(f"Books: {total:,}\n")

    print("Description statistics")
    print("----------------------")
    if total > 0:
        print(f"Mean chars : {sum(lengths)/total:.1f}")
        print(f"Max chars  : {max(lengths):,}")
        print(f"Mean bytes : {sum(byte_lengths)/total:.1f}")
        print(f"Max bytes  : {max(byte_lengths):,}")
    else:
        print("Mean chars : N/A")
        print("Max chars  : N/A")
        print("Mean bytes : N/A")
        print("Max bytes  : N/A")

    print()
    print("Potential issues")
    print("----------------")
    print(f"HTML tags           : {len(html_books):,}")
    print(f"HTML entities       : {len(entity_books):,}")
    print(f"Control chars (Cc)  : {len(control_books):,}")
    print(f"Format chars (Cf)   : {len(format_books):,}")

    print()
    print("Top Unicode script prefixes")
    print("---------------------------")
    for name, count in script_counter.most_common(20):
        print(f"{name:<15} {count:,}")

    print()
    print("Books containing HTML")
    print("---------------------")
    for legacy_id, title in html_books[:25]:
        print(f"{legacy_id:>10}  {title}")

    print()
    print("Books containing control characters")
    print("-----------------------------------")
    for legacy_id, title, count in control_books[:25]:
        print(f"{legacy_id:>10}  ({count})  {title}")

    print()
    print("Books containing format characters")
    print("----------------------------------")
    for legacy_id, title, count in format_books[:25]:
        print(f"{legacy_id:>10}  ({count})  {title}")

    print()
    print("Sample HTML descriptions")
    print("------------------------")
    for legacy_id, title in html_books[:10]:
        desc = next(d for i, t, d in rows if i == legacy_id)
        print()
        print(f"{legacy_id} — {title}")
        print(repr(desc[:300]))


if __name__ == "__main__":
    main()
