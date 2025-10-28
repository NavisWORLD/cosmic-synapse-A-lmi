"""
Global Crawler Subsystem
Ethical web crawling with Scrapy and Selenium/Playwright
"""

import scrapy
import logging
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.http import Request, Response
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)


class GlobalCrawler(sprapy.Spider):
    """
    Ethical web crawler for A-LMI system.
    
    Features:
    - Respects robots.txt
    - Handles JavaScript-heavy sites via Selenium
    - Publishes to Kafka EventBus
    - Extracts text, images, and structured data
    """
    
    name = 'global_crawler'
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_DELAY': 0.5,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'USER_AGENT': 'A-LMI-Bot/2.0 (+https://example.com/bot)',
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selenium_enabled = kwargs.get('selenium', True)
        self.kafka_producer = kwargs.get('kafka_producer', None)
        
    def start_requests(self):
        """Generate initial requests."""
        start_urls = getattr(self, 'start_urls', [])
        
        for url in start_urls:
            yield Request(url=url, callback=self.parse, dont_filter=False)
    
    def parse(self, response: Response):
        """
        Parse webpage and extract content.
        
        Args:
            response: Scrapy response object
        """
        page_url = response.url
        
        # Extract text content
        text_content = response.css('body *::text').getall()
        text_content = ' '.join(text_content).strip()
        
        # Extract images
        images = response.css('img::attr(src)').getall()
        images = [response.urljoin(img) for img in images]
        
        # Extract links
        links = response.css('a::attr(href)').getall()
        links = [response.urljoin(link) for link in links if link]
        
        # Extract metadata
        title = response.css('title::text').get()
        meta_description = response.css('meta[name="description"]::attr(content)').get()
        
        # Publish to Kafka
        if self.kafka_producer:
            event = {
                "event_type": "raw.web.content",
                "url": page_url,
                "title": title,
                "meta_description": meta_description,
                "text_content": text_content,
                "images": images,
                "links": links,
                "timestamp": response.meta.get('download_latency', 0)
            }
            
            self.kafka_producer.publish("raw.web.content", event)
            logger.info(f"Crawled and published: {page_url}")
        
        # Follow links (limit depth)
        depth = response.meta.get('depth', 0)
        max_depth = getattr(self, 'max_depth', 2)
        
        if depth < max_depth:
            for link in links[:10]:  # Limit links to follow
                yield Request(url=link, callback=self.parse, meta={'depth': depth + 1})
    
    def parse_javascript_page(self, url: str):
        """
        Parse JavaScript-heavy pages using Selenium.
        
        Args:
            url: URL to parse
        """
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extract content
            title = driver.title
            text_content = driver.find_element(By.TAG_NAME, "body").text
            
            # Extract images
            images = [img.get_attribute('src') for img in driver.find_elements(By.TAG_NAME, 'img')]
            
            driver.quit()
            
            return {
                "event_type": "raw.web.content",
                "url": url,
                "title": title,
                "text_content": text_content,
                "images": images,
                "javascript_rendered": True
            }
        except Exception as e:
            logger.error(f"Error parsing JavaScript page {url}: {e}")
            driver.quit()
            return None
    
    def closed(self, reason):
        """Called when crawler is closed."""
        logger.info(f"Crawler closed: {reason}")


def run_crawler(start_urls: list, kafka_producer=None, selenium=True):
    """
    Run the global crawler.
    
    Args:
        start_urls: List of starting URLs
        kafka_producer: Kafka producer instance
        selenium: Enable Selenium for JavaScript pages
    """
    settings = get_project_settings()
    
    process = CrawlerProcess(settings)
    process.crawl(
        GlobalCrawler,
        start_urls=start_urls,
        selenium=selenium,
        kafka_producer=kafka_producer
    )
    process.start()

