from bs4 import BeautifulSoup
import requests
import re
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import time

from dotenv import load_dotenv
import os

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess

CRAWL_DELAY = 15 # crawl-delay stated by robots.txt
ALLOWED_PATHS = ["/archive", "year", "/abs", "/list", "/pdf", "/html", "/catchup"]
VALID_YEARS = [str(i) for i in range(2015, 2024)]

class ArxivCrawler(scrapy.spiders.CrawlSpider):
    name = "arxiv_crawler"
    custom_settings = {
        "DOWNLOAD_DELAY": CRAWL_DELAY,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 15,
        "AUTOTHROTTLE_MAX_DELAY": 60,
    }
    crawl_delay = CRAWL_DELAY
    # allowed_domains = ["arxiv.org"]
    # start_urls = ["https://arxiv.org/archive"]

    rules = (
        # step 1: find categories within subjects -> get to lists of recent articles
        Rule(LinkExtractor(allow=(r"/list/.+/recent",), deny=(r"\?.*")), callback="log_url"),
        # step 2: find recent articles within categories -> get to abstracts
        Rule(LinkExtractor(allow=(r"(/abs/\d{4}\.\d{4,5})|(/abs/[a-z\-]+/\d{7})",)), callback="parse_abstract"),
    )
    
    def __init__(self, category=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if category:
            self.start_urls = [f"https://arxiv.org/archive/{category}"]
            self.allowed_domains = ["arxiv.org"]
        # self.start_urls = [f"https://arxiv.org/list/cs.AI/recent"]
    
    def open_spider(self, spider):
        atlas_uri = os.getenv("ATLAS_URI")
        self.client = MongoClient(atlas_uri)
        self.collection = self.client["arxiv_db"]["arxiv_collection"]
        
    def close_spider(self, spider):
        self.client.close()
        
    def log_url(self, response: scrapy.http.Response):
        self.logger.info(f"Retrieved URL: {response.url}")

    def parse_abstract(self, response: scrapy.http.Response):
        title = response.css("h1.title::text").getall()
        abstract = response.css("blockquote.abstract::text").getall()
        self.logger.info(f"Retrieved title: {title}")
        self.logger.info(f"Retrieved abstract: {abstract[:50]} ...")

        document = {
            "url": response.url,
            "title": "".join(title).strip(),
            "abstract": "".join(abstract).strip()
        }

        # insert data into Atlas Collection
        self.collection.insert_one(document)

        yield document

    