import argparse
import json
import re
import sqlite3
import unicodedata
from collections import Counter

import pandas as pd

HTML_TAG_RE = re.compile(r'<\s*([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>')
ENTITY_RE = re.compile(r'&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
MULTI_SPACE_RE = re.compile(r'[ \t]{2,}')
MULTI_NEWLINE_RE = re.compile(r'\n{2,}')
MOJIBAKE_MARKERS = ('Ã', 'â€', 'Â', 'ð', '\ufffd')  # \ufffd = U+FFFD replacement char


def check_raw_nul_bytes(path):
    """Warn if the raw CSV file contains NUL bytes: pandas' CSV parser silently
    truncates strings at NUL, which would otherwise hide this issue entirely."""
    if not path.endswith('.csv'):
        return None
    with open(path, 'rb') as f:
        raw = f.read()
    return raw.count(b'\x00')


def load_data(path, table=None, column='description', query=None):
    if path.endswith(('.db', '.sqlite', '.sqlite3')):
        conn = sqlite3.connect(path)
        try:
            if query:
                df = pd.read_sql_query(query, conn)
            else:
                if not table:
                    raise ValueError("Must supply --table for sqlite input unless --query is given")
                df = pd.read_sql_query(f"SELECT {column} FROM {table}", conn)
        finally:
            conn.close()
    else:
        df = pd.read_csv(path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
    return df


def analyze_text(text):
    """Return a dict of findings for a single description string."""
    if text is None or (isinstance(text, float)):  # NaN from pandas
        return {'is_null': True}

    findings = {'is_null': False, 'length': len(text)}

    findings['html_tags'] = HTML_TAG_RE.findall(text)
    findings['html_entities'] = ENTITY_RE.findall(text)

    control_chars, format_chars, other_chars = [], [], []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == 'Cc' and ch not in ('\n', '\t'):
            control_chars.append(ch)
        elif cat == 'Cf':
            format_chars.append(ch)
        elif cat.startswith('C'):
            other_chars.append(ch)
    findings['control_chars'] = control_chars
    findings['format_chars'] = format_chars
    findings['other_control_category_chars'] = other_chars

    findings['has_multi_space'] = bool(MULTI_SPACE_RE.search(text))
    findings['has_multi_newline'] = bool(MULTI_NEWLINE_RE.search(text))
    findings['leading_trailing_ws'] = text != text.strip()
    findings['urls'] = URL_RE.findall(text)

    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    findings['non_ascii_ratio'] = non_ascii / len(text) if text else 0
    findings['possible_mojibake'] = any(m in text for m in MOJIBAKE_MARKERS)
    findings['nfkc_changes_text'] = unicodedata.normalize('NFKC', text) != text

    return findings


def aggregate(df, column, sample_n=3):
    total = len(df)
    stats = Counter()
    tag_counter, entity_counter = Counter(), Counter()
    control_char_counter, format_char_counter = Counter(), Counter()
    lengths = []
    examples = {k: [] for k in
                ['html_tags', 'control_chars', 'format_chars', 'mojibake', 'multi_space', 'urls']}

    for val in df[column]:
        f = analyze_text(val)
        if f.get('is_null'):
            stats['null_or_empty'] += 1
            continue

        lengths.append(f['length'])

        if f['html_tags']:
            stats['contains_html_tags'] += 1
            tag_counter.update(t.lower() for t in f['html_tags'])
            if len(examples['html_tags']) < sample_n:
                examples['html_tags'].append(val[:200])
        if f['html_entities']:
            stats['contains_html_entities'] += 1
            entity_counter.update(f['html_entities'])
        if f['control_chars']:
            stats['contains_control_chars'] += 1
            control_char_counter.update(f['control_chars'])
            if len(examples['control_chars']) < sample_n:
                examples['control_chars'].append(repr(val[:200]))
        if f['format_chars']:
            stats['contains_format_chars'] += 1
            format_char_counter.update(f['format_chars'])
            if len(examples['format_chars']) < sample_n:
                examples['format_chars'].append(repr(val[:200]))
        if f['other_control_category_chars']:
            stats['contains_other_control_category'] += 1
        if f['urls']:
            stats['contains_urls'] += 1
            if len(examples['urls']) < sample_n:
                examples['urls'].append(val[:200])
        if f['has_multi_space']:
            stats['has_multi_space'] += 1
            if len(examples['multi_space']) < sample_n:
                examples['multi_space'].append(repr(val[:200]))
        if f['has_multi_newline']:
            stats['has_multi_newline'] += 1
        if f['leading_trailing_ws']:
            stats['leading_trailing_ws'] += 1
        if f['possible_mojibake']:
            stats['possible_mojibake'] += 1
            if len(examples['mojibake']) < sample_n:
                examples['mojibake'].append(val[:200])
        if f['nfkc_changes_text']:
            stats['nfkc_changes_text'] += 1

    length_stats = {}
    if lengths:
        ls = sorted(lengths)
        n = len(ls)
        length_stats = {'min': ls[0], 'max': ls[-1], 'mean': sum(ls) / n, 'median': ls[n // 2]}

    return {
        'total_rows': total,
        'stats': dict(stats),
        'length_stats': length_stats,
        'top_tags': tag_counter.most_common(20),
        'top_entities': entity_counter.most_common(20),
        'top_control_chars': [(repr(c), n) for c, n in control_char_counter.most_common(20)],
        'top_format_chars': [(repr(c), n) for c, n in format_char_counter.most_common(20)],
        'examples': examples,
    }


def print_report(result):
    total = result['total_rows']
    stats = result['stats']

    def pct(key):
        n = stats.get(key, 0)
        return f"{n} ({n / total * 100:.1f}%)" if total else str(n)

    print("=" * 70)
    print("DESCRIPTION COLUMN DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Total rows: {total}")
    print(f"Null/empty: {pct('null_or_empty')}\n")

    ls = result['length_stats']
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

    if result['top_tags']:
        print("-- Most common HTML tags --")
        for tag, n in result['top_tags']:
            print(f"  <{tag}>: {n}")
        print()

    if result['top_entities']:
        print("-- Most common HTML entities --")
        for ent, n in result['top_entities']:
            print(f"  &{ent};: {n}")
        print()

    if result['top_control_chars']:
        print("-- Most common control characters (Cc) --")
        for c, n in result['top_control_chars']:
            print(f"  {c}: {n}")
        print()

    if result['top_format_chars']:
        print("-- Most common format characters (Cf: zero-width/bidi/etc.) --")
        for c, n in result['top_format_chars']:
            print(f"  {c}: {n}")
        print()

    print("-- Sample offending rows --")
    for key, samples in result['examples'].items():
        if samples:
            print(f"  [{key}]")
            for s in samples:
                print(f"    {s}")
    print()

    print_recommendations(stats)


def print_recommendations(stats):
    print("=" * 70)
    print("RECOMMENDED PREPROCESSING STEPS")
    print("=" * 70)
    recs = []
    if stats.get('contains_html_tags'):
        recs.append("Strip HTML tags, e.g. BeautifulSoup(text, 'html.parser').get_text() "
                     "(preferred over regex — handles malformed markup safely).")
    if stats.get('contains_html_entities'):
        recs.append("Unescape HTML entities with html.unescape() before/alongside tag stripping.")
    if stats.get('contains_control_chars') or stats.get('contains_other_control_category'):
        recs.append("Remove non-printable control characters, e.g. "
                     "''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\\n\\t').")
    if stats.get('contains_format_chars'):
        recs.append("Strip Unicode format characters (category 'Cf': zero-width spaces, joiners, "
                     "bidi marks) — they add no semantic value and can confuse tokenizers.")
    if stats.get('nfkc_changes_text'):
        recs.append("Apply unicodedata.normalize('NFKC', text) to fold compatibility characters "
                     "(fullwidth forms, ligatures, etc.) into standard forms.")
    if stats.get('possible_mojibake'):
        recs.append("Investigate mojibake/double-encoding (e.g. UTF-8 bytes decoded as Latin-1). "
                     "The 'ftfy' library's fix_text() is a good targeted fix; also check ingestion encoding.")
    if stats.get('has_multi_space') or stats.get('has_multi_newline') or stats.get('leading_trailing_ws'):
        recs.append("Normalize whitespace: collapse repeats and strip ends, "
                     "e.g. re.sub(r'\\s+', ' ', text).strip().")
    if stats.get('contains_urls'):
        recs.append("Decide whether to keep, mask, or drop URLs depending on whether they carry "
                     "semantic signal for your embedding use case.")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?', default='data/goodreads.db', help='Path to CSV file or SQLite .db/.sqlite file (default: data/goodreads.db)')
    parser.add_argument('--table', default='books', help='Table name (for SQLite input, default: books)')
    parser.add_argument('--column', default='description', help='Column to analyze (default: description)')
    parser.add_argument('--query', help='Custom SQL query for SQLite input (overrides --table)')
    parser.add_argument('--sample', type=int, default=3, help='Example rows to show per issue type')
    parser.add_argument('--json-out', help='Optional path to write full results as JSON')
    args = parser.parse_args()

    nul_count = check_raw_nul_bytes(args.path)
    if nul_count:
        print(f"WARNING: raw file contains {nul_count} NUL byte(s). pandas' CSV parser silently "
              f"truncates strings at NUL, so affected rows may show as shorter/cleaner than they "
              f"really are. Investigate the source export/encoding before trusting length stats "
              f"below.\n")

    df = load_data(args.path, table=args.table, column=args.column, query=args.query)
    result = aggregate(df, args.column, sample_n=args.sample)
    print_report(result)

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull results written to {args.json_out}")


if __name__ == '__main__':
    main()
