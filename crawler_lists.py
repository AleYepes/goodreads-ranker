import re
import csv
import time
import os
from urllib.parse import urljoin
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Format: https://www.goodreads.com/review/list/75706676?print=true&sort=date_read&view=reviews
LIST_IDS = [
    104343033, #104343033-santiago-lecumberri **
    104945614, #104945614-aadil-kumar
    105258888, #105258888-zach-saylor
    11116469, #11116469-sebastian-gebski
    113185438, #113185438-taiyr
    115764833, #115764833-austin-george
    118560285, #118560285-bichons-and-books-nz
    124720847, #124720847-abrish
    129155685, #129155685-ignacio-mu-oz-lanza **
    13448447, #13448447-lyda
    13647498, #13647498.Oana_David
    13737030, #13737030-maddy
    156484926, #156484926-victoria
    160516894, #160516894-janine
    166997642, #166997642-carson-cummins
    174792571, #174792571-annie
    1834894, #1834894.Manny_Rayner
    18922126, #18922126-ella-park
    18913667, #18913667.Edward_Vass
    21397146, #21397146-stefy
    22482559, #22482559-mathi-fonseca
    22726983, #22726983-gast-n-mousqu-s
    23161382, #23161382-vanesa
    24885719, #24885719-fran-oise
    267189, #267189-todd-n
    26052616, #26052616-margherita
    27115955, #27115955-catherine-wood
    22978411, #22978411-cristina
    31565140, #31565140-irina-toledo
    33074940, #33074940-anca-e-milea
    34518408, #34518408-dawood
    40426330, #40426330-sabrina-li
    41797321, #41797321-cristina-cojocaru
    42001957, #42001957-matty-van-hoof
    43400637, #43400637-fay-pretty
    46459461, #46459461-an-fech
    51281420, #51281420-daniela-g-mez
    54115664, #54115664-mandy
    5868084, #5868084-mairi
    65139494, #65139494-daniel-castro
    70012245, #70012245-till-chen
    7043947, #7043947-andra-enache
    75706676, #75706676-steve-abreu
    76860332, #76860332-cecilia
    8136076, #8136076.Cosmin_Leucu_a
    90649237, #90649237-sara
    91998392, #91998392-daniel-prelipcean
    ]

OUTPUT_FILE = "data/friend_ratings.csv"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SIGNIN_URL = "https://www.goodreads.com/ap/signin?language=en_US&openid.assoc_handle=amzn_goodreads_web_na&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in"
LIST_SORT = "date_added"
LIST_ORDER = "d"
FIELDNAMES = ["list_id", "book_id", "title", "rating", "num_pages", "date_read", "date_added"]

RATING_MAP = {
    "it was amazing": 5,
    "really liked it": 4,
    "liked it": 3,
    "it was ok": 2,
    "did not like it": 1,
}

def clean_text(text):
    if text:
        return text.strip().replace("\n", "")
    return ""

def is_login_page(page):
    return (
        "/ap/signin" in page.url
        or page.locator('input[type="email"]').count() > 0
        or page.locator("button.authPortalSignInButton").count() > 0
    )

def wait_for_post_login(page):
    for selector in ("#booksBody", "#books", "#reviewPagination", ".homePrimaryColumn"):
        try:
            page.wait_for_selector(selector, timeout=15000)
            return
        except PlaywrightTimeoutError:
            continue

    if is_login_page(page):
        raise RuntimeError("Goodreads login did not complete successfully.")

def login_to_goodreads(page, email, password):
    if not email or not password:
        raise RuntimeError("GOODREADS_EMAIL and GOODREADS_PASSWORD must be set to scrape private review lists.")

    if page.locator("button.authPortalSignInButton").count() > 0:
        with page.expect_navigation(wait_until="domcontentloaded"):
            page.locator("button.authPortalSignInButton").click()
    elif page.locator('input[type="email"]').count() == 0:
        page.goto(SIGNIN_URL, wait_until="domcontentloaded")

    page.wait_for_selector('input[type="email"]', timeout=30000)
    page.fill('input[type="email"]', email)
    page.fill('input[type="password"]', password)
    page.click('input[type="submit"]')
    page.wait_for_load_state("domcontentloaded")
    wait_for_post_login(page)

def open_list_page(page, list_id, email, password):
    url = f"https://www.goodreads.com/review/list/{list_id}?print=true&sort={LIST_SORT}&order={LIST_ORDER}&view=reviews"
    page.goto(url, wait_until="domcontentloaded")

    if is_login_page(page):
        print("Login required. Authenticating session...")
        login_to_goodreads(page, email, password)
        page.goto(url, wait_until="domcontentloaded")

    return url

def ensure_list_page(page, target_url, email, password):
    if is_login_page(page):
        print("Login required. Authenticating session...")
        login_to_goodreads(page, email, password)
        page.goto(target_url, wait_until="domcontentloaded")

def load_existing_rows():
    rows_by_key = {}
    if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
        return rows_by_key

    with open(OUTPUT_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            list_id = str(row.get("list_id", "")).strip()
            book_id = str(row.get("book_id", "")).strip()
            if not list_id or not book_id:
                continue
            normalized = {field: str(row.get(field, "")).strip() for field in FIELDNAMES}
            rows_by_key[(list_id, book_id)] = normalized
    return rows_by_key

def save_rows(rows_by_key):
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_by_key.values())

def extract_row(row, list_id):
    title_el = row.query_selector(".field.title a")
    title = clean_text(title_el.inner_text()) if title_el else "Unknown"

    href = title_el.get_attribute("href") if title_el else ""
    book_id_match = re.search(r'/book/show/(\d+)', href)
    book_id = book_id_match.group(1) if book_id_match else ""
    if not book_id:
        return None

    rating_el = row.query_selector(".field.rating .staticStars")
    rating_text = rating_el.get_attribute("title") if rating_el else ""
    rating = str(RATING_MAP.get(rating_text, 0))

    pages_el = row.query_selector(".field.num_pages .value")
    raw_pages = pages_el.text_content() if pages_el else ""
    num_pages = re.sub(r"[^\d]", "", raw_pages)

    date_read_el = row.query_selector(".field.date_read .date_read_value")
    date_read = clean_text(date_read_el.inner_text()) if date_read_el else ""

    date_added_el = row.query_selector(".field.date_added span")
    date_added = ""
    if date_added_el:
        date_added = date_added_el.get_attribute("title")
        if not date_added:
            date_added = clean_text(date_added_el.inner_text())

    return {
        "list_id": str(list_id),
        "book_id": book_id,
        "title": title,
        "rating": rating,
        "num_pages": num_pages,
        "date_read": date_read,
        "date_added": date_added,
    }

def scrape_goodreads():
    load_dotenv()
    email = os.getenv("GOODREADS_EMAIL")
    password = os.getenv("GOODREADS_PASSWORD")
    rows_by_key = load_existing_rows()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent=USER_AGENT
        )
        page = context.new_page()

        for list_id in LIST_IDS:
            print(f"Scraping {list_id}")
            page_num = 1
            current_url = open_list_page(page, list_id, email, password)
            list_changed = False

            while True:
                try:
                    ensure_list_page(page, current_url, email, password)
                    page.wait_for_selector("#booksBody", timeout=10000)
                except Exception:
                    print(f"Could not find book table for {list_id}. Skipping or finished.")
                    break

                rows = page.query_selector_all("tr.bookalike.review")
                page_all_known = True

                for row in rows:
                    try:
                        extracted = extract_row(row, list_id)
                        if not extracted:
                            continue

                        key = (extracted["list_id"], extracted["book_id"])
                        existing = rows_by_key.get(key)
                        if existing != extracted:
                            rows_by_key[key] = extracted
                            list_changed = True
                            page_all_known = False
                        elif existing is None:
                            page_all_known = False
                    except Exception as e:
                        print(f"Error extracting row: {e}")
                        page_all_known = False
                        continue

                if page_all_known:
                    print(f"    P{page_num} unchanged, stopping early")
                    break

                next_button = page.query_selector("a.next_page")
                next_class = next_button.get_attribute("class") if next_button else ""
                next_href = next_button.get_attribute("href") if next_button else None
                if next_button and next_href and "disabled" not in next_class:
                    current_url = urljoin(page.url, next_href)
                    with page.expect_navigation():
                        next_button.click()
                    print(f'    P{page_num}')
                    page_num += 1
                    time.sleep(1)
                else:
                    print(f'Finished {list_id}')
                    break

            if list_changed:
                save_rows(rows_by_key)

        browser.close()

    if not os.path.exists(OUTPUT_FILE):
        save_rows(rows_by_key)

if __name__ == "__main__":
    scrape_goodreads()
