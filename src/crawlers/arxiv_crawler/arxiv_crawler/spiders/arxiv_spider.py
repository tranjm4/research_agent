from pymongo.mongo_client import MongoClient

from dotenv import load_dotenv
import os

import logging
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

CRAWL_DELAY = 15 # crawl-delay stated by robots.txt
ALLOWED_PATHS = ["/archive", "year", "/abs", "/list", "/pdf", "/html", "/catchup"]
VALID_YEARS = [str(i) for i in range(2015, 2024)]

class ArxivCrawler(CrawlSpider):
    name = "arxiv_crawler"
    custom_settings = {
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "%(levelname)s: %(message)s",
        "LOG_ENABLED": True,
        "LOG_DISABLED": [
            "pymongo.topology",
            "pymongo.connection",
            "scrapy.core.engine",
            "scrapy.middleware",
            "scrapy.extensions.logstats",
        ],
        
        "DOWNLOAD_DELAY": CRAWL_DELAY,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 15,
        "AUTOTHROTTLE_MAX_DELAY": 60,
    }
    DISABLED_LOGS = [
        "pymongo.topology",
        "pymongo.connection",
        "scrapy.core.engine",
        "scrapy.middleware",
        "scrapy.extensions.logstats",
        "scrapy.addons",
        "scrapy.extensions.telnet",
    ]
    
    crawl_delay = CRAWL_DELAY
    # allowed_domains = ["arxiv.org"]
    # start_urls = ["https://arxiv.org/archive"]

    rules = (
        # step 1: find categories within subjects -> get to lists of recent articles
        Rule(LinkExtractor(allow=(r"/list/.+/recent",), deny=(r"\?.*")), callback="log_url", follow=True),
        # step 2: find recent articles within categories -> get to abstracts
        Rule(LinkExtractor(allow=(r"(/abs/\d{4}\.\d{4,5})|(/abs/[a-z\-]+/\d{7})",)), callback="parse_abstract", follow=True),
    )
    
    def __init__(self, category=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if category:
            self.start_urls = [f"https://arxiv.org/archive/{category}"]
            self.allowed_domains = ["arxiv.org"]
        # self.client = self.connect_atlas_client()
        # self.collection = self.client["arxiv_db"]["arxiv_collection"]
        
        self.configure_log_settings()
    
    def configure_log_settings(self):
        for log in ArxivCrawler.DISABLED_LOGS:
            logger = logging.getLogger(log)
            logger.setLevel(logging.WARNING)
            
        logging.basicConfig(
            filename="log.txt", level=logging.INFO
        )
        
    
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(ArxivCrawler, cls).from_crawler(crawler, *args, **kwargs)
        spider.client = spider.connect_atlas_client()
        spider.collection = spider.client["arxiv_db"]["arxiv_collection"]
        return spider
    
    def connect_atlas_client(self):
        self.logger.info("open_spider() called - initializing MongoDB connection")
        load_dotenv()
        atlas_uri = os.getenv("ATLAS_URI")
        if not atlas_uri:
            self.logger.error("ENV VARIABLE ERROR: ATLAS_URI not found in the environment")
        client = MongoClient(atlas_uri)
        try:
            client.admin.command("ping")
        except:
            self.logger.error("CONNECTION ERROR: Unable to establish MongoDB connection")
        
        self.logger.info("MongoDB connection established")
        
        return client

    def close_spider(self, spider):
        if hasattr(self, 'client'):
            self.client.close()
        
    def log_url(self, response: scrapy.http.Response):
        self.logger.info(f"Retrieved URL: {response.url}")

    def parse_abstract(self, response: scrapy.http.Response):
        title = response.css("h1.title::text").getall()
        abstract = response.css("blockquote.abstract::text").getall()
        
        title = "".join(title).strip()
        abstract = "".join(abstract).strip()
        
        self.logger.info(f"Retrieved title: {title}")
        self.logger.info(f"Retrieved abstract: {abstract[:50]} ...")

        document = {
            "url": response.url,
            "title": title,
            "abstract": abstract
        }

        # insert data into Atlas Collection
        self.collection.insert_one(document)

        yield document

    