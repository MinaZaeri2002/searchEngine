import json
import time
import re
import os
import requests
import random
import hashlib
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from urllib import robotparser
from collections import deque
import logging
import threading

class AdvancedCrawler:
    def __init__(self, start_url, max_pages=4000, delay=1):
        self.start_url = self._normalize_url(start_url)
        self.max_pages = max_pages
        self.delay = delay
        self.domain = urlparse(self.start_url).netloc
        self.url_queue = deque([self.start_url])
        self.visited = set()
        self.duplicate_count = 0
        self.error_count = 0
        self.crawled_data = {}
        self.crawl_stats = {
            'total_crawled': 0,
            'final_queue_length': 0,
            'duplicate_urls': 0,
            'error_count': 0,
            'crawl_time': 0,
            'data_file_size_bytes': 0
        }
        self._setup_logging()
        self._setup_session()
        self._fetch_robots_txt()

    def _url_hash(self, url):
        return hashlib.md5(url.encode()).hexdigest()

    def _normalize_url(self, url):
        parsed = urlparse(url)
        path = parsed.path.rstrip('/') or '/'
        return parsed._replace(path=path, query="").geturl().lower()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('crawler.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_session(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
            ])
        })

    def _rotate_user_agent(self):
        self.session.headers.update({'User-Agent': random.choice([
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0'
        ])})

    def _fetch_robots_txt(self):
        robots_url = urljoin(self.start_url, '/robots.txt')
        try:
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 200:
                self.robot_parser = robotparser.RobotFileParser()
                self.robot_parser.set_url(robots_url)
                self.robot_parser.read()
            else:
                self.robot_parser = None
        except Exception as e:
            self.logger.error(f"Robots.txt fetch failed: {str(e)}")
            self.robot_parser = None

    def _is_allowed(self, url):
        return self.robot_parser.can_fetch('*', url) if self.robot_parser else True

    def _is_valid(self, url):
        parsed_url = urlparse(url)
        return (parsed_url.netloc == self.domain and
                parsed_url.scheme in ('http', 'https') and
                not re.search(r'\.(jpg|jpeg|png|gif|pdf|zip)$', url))

    def _normalize_text(self, text):
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_content(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.text.strip() if soup.title else ""
        title = self._normalize_text(title)

        body = soup.find('body')
        body_text = body.get_text(separator=" ", strip=True) if body else ""
        body_text = self._normalize_text(body_text)

        return title, body_text

    def _handle_links(self, soup, base_url):
        new_links = 0
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(base_url, link['href'])
            normalized_url = self._normalize_url(absolute_url)

            if normalized_url in self.visited:
                self.duplicate_count += 1
                continue

            if self._is_valid(normalized_url):
                self.url_queue.append(normalized_url)
                new_links += 1
        return new_links

    def _save_output(self):
        with open('crawled_pages.json', 'w', encoding='utf-8') as f:
            json.dump(self.crawled_data, f, ensure_ascii=False, indent=4)

        self.crawl_stats['duplicate_urls'] = self.duplicate_count
        self.crawl_stats['error_count'] = self.error_count

        try:
            file_size = os.path.getsize('crawled_pages.json')
        except OSError:
            file_size = 0
        self.crawl_stats['data_file_size_bytes'] = file_size

        report = {
            'statistics': self.crawl_stats
        }

        with open('crawl_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)

    def _status_monitor(self):
        while True:
            print(
                f"\rCrawled: {self.crawl_stats['total_crawled']}/{self.max_pages} | Queue: {len(self.url_queue)} | Errors: {self.error_count}",
                end='')
            time.sleep(1)

    def crawl(self):
        monitor_thread = threading.Thread(target=self._status_monitor, daemon=True)
        monitor_thread.start()

        start_time = time.time()

        try:
            while self.crawl_stats['total_crawled'] < self.max_pages and self.url_queue:
                current_url = self.url_queue.popleft()

                if current_url in self.visited:
                    continue
                self.visited.add(current_url)

                if not self._is_allowed(current_url):
                    continue

                try:
                    self._rotate_user_agent()
                    time.sleep(random.uniform(self.delay, self.delay * 1.5))

                    response = self.session.get(current_url, timeout=15)
                    if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                        title, body_text = self._extract_content(response.text)

                        url_hash = self._url_hash(current_url)
                        self.crawled_data[url_hash] = {
                            'url': current_url,
                            'title': title,
                            'body': body_text
                        }

                        soup = BeautifulSoup(response.text, 'html.parser')
                        new_links = self._handle_links(soup, current_url)

                        self.crawl_stats['total_crawled'] += 1
                        self.logger.info(f"Crawled: {current_url} | New links: {new_links}")

                except Exception as e:
                    self.error_count += 1
                    self.logger.error(f"Error crawling {current_url}: {str(e)}")

        finally:
            end_time = time.time()
            self.crawl_stats['crawl_time'] = round(end_time - start_time, 2)
            self.crawl_stats['final_queue_length'] = len(self.url_queue)
            self.crawl_stats['error_count'] = self.error_count
            self._save_output()
            print(f"\n\nCrawl completed in {self.crawl_stats['crawl_time']} seconds")
            print(f"Final stats: {json.dumps(self.crawl_stats, indent=4)}")

if __name__ == "__main__":
    crawler = AdvancedCrawler(
        start_url="https://www.zoomit.ir/",
        max_pages=4000,
        delay=1
    )
    crawler.crawl()