"""
lab1 phase 1 - ethical web crawler
crawls urls, checks robots.txt, extracts text with trafilatura,
keeps only pages with at least 500 words, saves to jsonl
"""

import json
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from typing import List, Dict, Optional

import httpx
import trafilatura


class WebCrawler:
    """ethical crawler that respects robots.txt and extracts main content"""

    def __init__(self, output_file: str = "crawler_output.jsonl", min_word_count: int = 500):
        self.output_file = output_file
        self.visited_urls = set()
        self.min_word_count = min_word_count

        # use a browser-like user agent so we don't get blocked
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

        self.client = httpx.Client(
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            follow_redirects=True,
            timeout=httpx.Timeout(25.0),
            http2=False,
        )

        # cache robots parsers per domain so we don't fetch robots.txt twice
        self.robots_cache: Dict[str, RobotFileParser] = {}

    def _get_robot_parser(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain in self.robots_cache:
            return self.robots_cache[domain]

        robots_url = f"{parsed.scheme}://{domain}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)

        try:
            r = self.client.get(robots_url)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            else:
                # if we can't reach robots.txt, assume crawling is allowed
                rp.parse([])
        except Exception:
            rp.parse([])

        self.robots_cache[domain] = rp
        return rp

    def check_robots_txt(self, url: str) -> bool:
        rp = self._get_robot_parser(url)
        return rp.can_fetch(self.user_agent, url)

    def is_useful_content(self, text: str) -> bool:
        if not text:
            return False
        return len(text.split()) >= self.min_word_count

    def extract_content(self, url: str, retries: int = 2) -> Optional[Dict]:

        if url in self.visited_urls:
            print(f"already visited: {url}")
            return None

        if not self.check_robots_txt(url):
            print(f" robots.txt not allowed: {url}")
            return None

        for attempt in range(1, retries + 2):
            try:
                print(f"downloading (attempt {attempt}): {url}")

                # trafilatura.fetch_url is more robust than plain httpx for html extraction
                downloaded = trafilatura.fetch_url(url)

                # fall back to httpx if trafilatura couldn't get the page
                if not downloaded:
                    resp = self.client.get(url)
                    if resp.status_code != 200:
                        print(f" HTTP {resp.status_code}: {url}")
                        continue
                    downloaded = resp.text

                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                    favor_recall=True
                )

                if not self.is_useful_content(text):
                    wc = len(text.split()) if text else 0
                    print(f"not enough content ({wc} words): {url}")
                    return None

                metadata = trafilatura.extract_metadata(downloaded)

                self.visited_urls.add(url)

                result = {
                    "url": url,
                    "text": text,
                    "word_count": len(text.split()),
                    "title": metadata.title if metadata else None,
                    "author": metadata.author if metadata else None,
                    "date": metadata.date if metadata else None,
                    "domain": urlparse(url).netloc,
                }

                print(f"Extracted: {result['word_count']} words| {result['title']}")
                return result

            except Exception as e:
                print(f"error {type(e).__name__}: {e}")
                if attempt < retries + 1:
                    time.sleep(1.0)
                else:
                    return None

        return None

    def save_to_jsonl(self, data: Dict):
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def crawl_urls(self, seed_urls: List[str], delay: float = 2.0):
        print(f"\nCrawling starting {len(seed_urls)} URLs")
        print(f"output: {self.output_file}")
        print(f"min word count: {self.min_word_count}\n")

        # reset output file
        Path(self.output_file).write_text("", encoding="utf-8")

        successful = 0

        for i, url in enumerate(seed_urls, 1):
            print(f"\n[{i}/{len(seed_urls)}] {url}")

            result = self.extract_content(url)

            if result:
                self.save_to_jsonl(result)
                successful += 1

            if i < len(seed_urls):
                print(f"Waiting {delay}s...")
                time.sleep(delay)

        print(f"\nFinished crawling: {successful}/{len(seed_urls)} pages OK")
        print(f"output file: {self.output_file}")

        self.client.close()


if __name__ == "__main__":
    seed_urls = [
        "https://en.wikipedia.org/wiki/Apollo_11",
        "https://en.wikipedia.org/wiki/Hubble_Space_Telescope",
        "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
        "https://en.wikipedia.org/wiki/NASA",
        "https://en.wikipedia.org/wiki/European_Space_Agency",
        "https://en.wikipedia.org/wiki/International_Space_Station",
        "https://en.wikipedia.org/wiki/Neil_Armstrong",
        "https://en.wikipedia.org/wiki/Space_exploration",
        "https://en.wikipedia.org/wiki/Ariane_(rocket_family)",
        "https://en.wikipedia.org/wiki/Artemis_program",
    ]

    crawler = WebCrawler(output_file="crawler_output.jsonl", min_word_count=500)
    crawler.crawl_urls(seed_urls, delay=2.0)

    print("\n" + "=" * 50)
    print("Preview")
    print("=" * 50 + "\n")

    try:
        with open("crawler_output.jsonl", "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                data = json.loads(line)
                print(f"{idx}. {data.get('title') or 'No title'}")
                print(f"   URL: {data['url']}")
                print(f"   Words: {data['word_count']}")
                print(f"   Domain: {data['domain']}\n")
    except FileNotFoundError:
        print("crawler_output.jsonl not found (crawling failed).")
