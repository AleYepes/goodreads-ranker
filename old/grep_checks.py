import html
import re
import sqlite3

DB_PATH = "data/goodreads.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT description FROM books WHERE description IS NOT NULL AND description != ''")
rows = cursor.fetchall()
conn.close()

descs = [r[0] for r in rows]
print(f"Total descriptions: {len(descs)}\n")

unclosed_tag_re = re.compile(r"<[a-zA-Z][^>]*$", re.MULTILINE)
unclosed_count = sum(1 for d in descs if unclosed_tag_re.search(d))
print(f"1a. Unclosed <letter... (no closing >) patterns: {unclosed_count}")

unclosed_with_space_re = re.compile(r"<[a-zA-Z][a-zA-Z0-9]* [^>\n]{0,100}(?=\n|$)")
unclosed_space_count = sum(1 for d in descs if unclosed_with_space_re.search(d))
print(f"1b. <tagname<space>content (potential unclosed tag): {unclosed_space_count}")
samples_1b = [d for d in descs if unclosed_with_space_re.search(d)][:3]
for i, s in enumerate(samples_1b):
    m = unclosed_with_space_re.search(s)
    print(f"  Sample {i+1} match: {repr(m.group()[:120])}")

print()

double_angle_re = re.compile(r"<<|>>")
double_angle_count = sum(1 for d in descs if double_angle_re.search(d))
print(f"2.  Descriptions containing << or >>: {double_angle_count}")
samples_2 = [d for d in descs if double_angle_re.search(d)][:5]
for i, s in enumerate(samples_2):
    print(f"  Sample {i+1}: {repr(s[:200])}")

print()

mso_re = re.compile(r"mso|<!--\[if|\[if gte|&lt;!--\[if", re.IGNORECASE)
mso_count = sum(1 for d in descs if mso_re.search(d))
print(f"3a. Raw descriptions with MSO/Word conditional comment patterns: {mso_count}")


def entity_decode(t: str) -> str:
    return html.unescape(t)


mso_decoded_re = re.compile(r"<!--\[if|<!--\s*\[if gte", re.IGNORECASE)
mso_decoded_count = sum(1 for d in descs if mso_decoded_re.search(entity_decode(d)))
print(f"3b. Descriptions with <!--[if after entity decode:              {mso_decoded_count}")

print("\n  MSO samples (raw[:300]):")
mso_samples = [d for d in descs if mso_re.search(d)][:5]
for i, s in enumerate(mso_samples):
    print(f"  Sample {i+1}: {repr(s[:300])}")
