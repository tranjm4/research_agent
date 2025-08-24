"""
HARVESTER

Interacts with arXiv's Open Archives Initiative (OAI) to harvest archive metadata.
"""

import requests
import xml.etree.ElementTree as ET

from tqdm import tqdm

import json
from datetime import datetime, timezone, timedelta
from time import sleep

from dotenv import load_dotenv
import os
from typing import Dict, Any, Optional
from typing_extensions import TypedDict

from concurrent.futures import ThreadPoolExecutor

from message_queue.message_queue import KafkaProducerWrapper

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

REQUEST_BASE_URL = "https://oaipmh.arxiv.org/oai?"

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}

TIMESTAMP_FILE = "/app/data/last_harvest.json"

TOPIC_NAME = os.getenv("TOPIC_NAME_PAPERS", "arxiv_papers")
        
class Document(TypedDict):
    url: str
    title: str
    authors: list[str]
    paper_id: str
    identifier: str
    datestamp: str
    created_date: str
    updated_date: str
    abstract: str
    topic: list[str]
    subtopics: list[str]
        
class Harvester:
    def __init__(self):
        self.producer = KafkaProducerWrapper(TOPIC_NAME)
        self.last_harvest_timestamp = None
    

    def send_request(self, resumption_token = None, from_date = None, until_date = None) -> requests.Response:
        """
        Sends a GET request to the arXiv OAI endpoint.
        Expects a resumption token if there are more records to receive.

        Returns:
            requests.Response: The HTTP response object from the arXiv request.
        """
        params = self._build_params(resumption_token, from_date, until_date)
        
        # make request
        response = requests.get(
            REQUEST_BASE_URL,
            params=params
        )
        
        return response

    
    def process_response(self, response: requests.Response, from_date, until_date) -> Optional[str]:
        """
        Given a 200 response, processes the XML and extracts relevant information.
        
        Args:
            response: The HTTP response object from the arXiv request
            from_date: The start date for the records
            until_date: The end date for the records
            
        Returns:
            Optional[str]: The resumption token for the next set of records, if available. Otherwise, None
        """
        root = ET.fromstring(response.text)
        
        # process response
        records = root.findall(".//oai:record", namespaces=NS)
        for record in tqdm(records, desc=f"Processing records {from_date} => {until_date}", unit="record"):
            document = self.extract_metadata(record)
            
            # Send the document to Kafka message queue (no key for better distribution)
            self.producer.send_message(document)
        
        resumption_token = root.find(".//oai:resumptionToken", namespaces=NS)
        if resumption_token is not None:
            resumption_token = resumption_token.text
        
        return resumption_token

    def extract_metadata(self, record):
        # Collect header information
        header = record.find("oai:header", NS)
        identifier = header.find("oai:identifier", NS).text
        datestamp = header.find("oai:datestamp", NS).text
        topics = header.findall("oai:setSpec", NS)
        
        # Collect topics and subtopics
        main_topics = list(set(topic.text.split(":")[0] for topic in topics))
        sub_topics = []
        for topic in topics:
            sub_topics.append(".".join(topic.text.split(":")[1:]))
        sub_topics = list(set(sub_topics))
        
        # Collect metadata information
        metadata = record.find("oai:metadata", NS)
        arxiv_meta = metadata.find("arxiv:arXiv", NS)
        
        title = arxiv_meta.find("arxiv:title", NS).text.strip()
        paper_id = arxiv_meta.find("arxiv:id", NS).text
        abstract = arxiv_meta.find("arxiv:abstract", NS).text.strip()
        created = arxiv_meta.find("arxiv:created", NS).text
        updated = arxiv_meta.find("arxiv:updated", NS)
        if updated:
            updated = updated.text
        else:
            updated = created
        
        authors = []
        for author in arxiv_meta.findall("arxiv:authors/arxiv:author", NS):
            keyname = author.find("arxiv:keyname", NS).text
            forenames = author.find("arxiv:forenames", NS)
            if forenames:
                forenames = forenames.text
                full_name = f"{forenames},{keyname}".strip()
            else:
                forenames = ""
                full_name = keyname.strip()
            authors.append(full_name)
            
        # Create document for Kafka message
        document = {
            "url": f"https://arxiv.org/abs/{paper_id}",
            "title": title,
            "authors": authors,
            "paper_id": paper_id,
            "identifier": identifier,
            "datestamp": datestamp,
            "created_date": created,
            "updated_date": updated,
            "abstract": abstract,
            "topic": main_topics,
            "subtopics": sub_topics,
        }
            
        return document

    def _build_params(self, resumption_token, from_date, until_date):
        params = {}
        if resumption_token:
            # do not include from and until parameters if resumption token
            params["resumptionToken"] = resumption_token
        else:
            params = {
                "verb": "ListRecords",
                "metadataPrefix": "arXiv"
            }
            if from_date:
                params["from"] = from_date
            if until_date:
                params["until"] = until_date
                
        return params
    
    def _load_last_timestamp(self) -> Optional[str]:
        """Loads the last timestamp from the persisted file
        
        If found, returns the last harvest timestamp as a string.
        Otherwise, returns None if file doesn't exist or key isn't defined
        
        Returns:
            str | None: The last harvest timestamp or None if not found.
        """
        try:
            with open(TIMESTAMP_FILE, 'r') as f:
                data = json.load(f)
                last_timestamp = data.get('last_harvest_timestamp')
                if last_timestamp:
                    logger.info(f"Using last harvest timestamp from file: {last_timestamp}")
                    return last_timestamp
                else:
                    logger.info("Key 'last_harvest_timestamp' not found in file")
                    return None
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.info(f"Could not load timestamp file: {e}. Starting fresh harvest")
            return None

    def _update_last_timestamp(self) -> None:
        """
        Updates the last harvest timestamp in the persisted file        
        """
        current_datetime_str = self._get_current_datetime()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(TIMESTAMP_FILE), exist_ok=True)
        
        # Save to file
        with open(TIMESTAMP_FILE, 'w') as f:
            json.dump({'last_harvest_timestamp': current_datetime_str}, f)
        
        logger.info(f"Updated last harvest timestamp in file: {current_datetime_str}")
        return

    def _get_current_datetime(self):
        current_utc = datetime.now(timezone.utc)
        current_utc_str = current_utc.strftime("%Y-%m-%d")
        
        return current_utc_str
    

    def _generate_week_ranges(self, start_date, end_date):
        current = start_date
        while current < end_date:
            week_end = min(current + timedelta(days=7), end_date)
            yield current.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")
            current = week_end

    def harvest(self, from_date, until_date):
        resumption_token = None
        logger.info(f"Starting harvest from {from_date} to {until_date}")
        while True:
            response = self.send_request(resumption_token=resumption_token, 
                                                from_date=from_date, 
                                                until_date=until_date)
            if response.status_code != 200:
                logger.error(f"Failed to retrieve data from arXiv: {response.status_code}")
                return
            
            # Valid 200 response
            resumption_token = self.process_response(response, from_date, until_date)
            
            if resumption_token is None:
                break
            sleep(3)
        logger.info(f"Harvest completed for {from_date} to {until_date}")
        sleep(3)
    
    def run(self):
        """Main entry point for the harvester"""
        try:
            logger.info("Starting harvester")
            end = datetime.now() # set end to today
            start = self._load_last_timestamp()
            if start == None:
                start = end - timedelta(days=365 * 4) # by default, set start point to 4 years ago
            else:
                # Convert string timestamp to datetime object
                start = datetime.strptime(start, "%Y-%m-%d")
                
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for from_str, until_str in self._generate_week_ranges(start, end):
                    futures.append(executor.submit(self.harvest, from_str, until_str))
                    
                for future in futures:
                    future.result()
                    
            # Update timestamp after successful harvest
            self._update_last_timestamp()
            logger.info("Harvest completed successfully")
            
        except Exception as e:
            logger.error(f"Harvest failed with error: {e}")
            raise
        finally:
            # Ensure Kafka producer is properly closed
            self.producer.close()
            logger.info("Kafka producer closed")


if __name__ == "__main__":
    harvester = Harvester()
    harvester.run() # TODO: incorporate cron job for periodic execution
