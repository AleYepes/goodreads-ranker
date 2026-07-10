"""
Test 4: httpx got a 202 (WAF challenge) even with fully valid auth cookies
(at-main, sess-at-main, x-main all present). That strongly suggests AWS WAF
Bot Control is fingerprinting the TLS handshake / HTTP2 behavior, not just
checking cookies+UA -- something httpx's stock TLS stack won't match no
matter what headers we add.

curl_cffi impersonates a real browser's TLS fingerprint (JA3/JA4) at the
socket level, which is the standard workaround for this exact WAF behavior.
This test reuses the cookies from session_cookies.json and see if that's
enough once the TLS fingerprint also looks legitimate.

Install first:
    pip install curl_cffi

Run:
    python 04_test_curl_cffi.py <list_id>
"""

import json
import sys

from curl_cffi import requests as cffi_requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def load_cookies(path="session_cookies.json"):
    with open(path) as f:
        raw = json.load(f)
    return {c["name"]: c["value"] for c in raw}


def main(list_id):
    cookies = load_cookies()
    url = f"https://www.goodreads.com/review/list/{list_id}?print=true&sort=date_added&order=d&view=reviews"

    # impersonate="chrome120" makes curl_cffi present Chrome's actual TLS
    # ClientHello / JA3 fingerprint, not just a matching User-Agent string.
    resp = cffi_requests.get(
        url,
        cookies=cookies,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        impersonate="chrome120",
        timeout=15,
    )

    print(f"Status: {resp.status_code}, final URL: {resp.url}")

    if resp.status_code == 202:
        print("Still 202 -- WAF is checking something curl_cffi's impersonation doesn't cover either.")
        print(resp.text[:500])
        return

    if "/ap/signin" in str(resp.url):
        print("Bounced to signin -- cookies expired, re-run 02_capture_session.py.")
        return

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("tr.bookalike.review")
    print(f"SUCCESS: Found {len(rows)} book rows on page 1")


if __name__ == "__main__":
    list_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not list_id:
        print("Usage: python 04_test_curl_cffi.py <list_id>")
        sys.exit(1)
    main(list_id)
