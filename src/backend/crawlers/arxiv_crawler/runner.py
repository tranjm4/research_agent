from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from arxiv_crawler.spiders.arxiv_spider import ArxivCrawler

categories = ["astro-ph", 
              "cond-mat", 
              "cs",
              "econ",
              "eess",
              "gr-qc",
              "hep-ex",
              "hep-lat",
              "hep-th",
              "math",
              "nlin",
              "nucl-ex",
              "nucl-th",
              "physics",
              "q-bio",
              "q-fin",
              "quant-ph",
              "stat"
             ]

process = CrawlerProcess(get_project_settings())

for cat in categories:
    process.crawl(ArxivCrawler, category=cat, atlas_collection="arxiv_categories")
    
process.start()