"""
PROCESSOR

Consumes messages from Kafka queue and processes arXiv paper data.
Handles PDF parsing, text extraction, and database insertion.
"""

import logging
from typing import Dict, Any, List, Optional
from kafka import KafkaConsumer
import signal
import sys
import fitz # PyMuPDF
import requests
import json
import time

from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure

from message_queue import KafkaConsumerWrapper

from dotenv import load_dotenv
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOPIC_NAME = os.getenv('TOPIC_NAME', 'arxiv_papers')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

class ArxivProcessor:
    def __init__(self):
        self.consumer = KafkaConsumerWrapper()
        self.mongo_client = MongoDBWrapper()
        self.running = True
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        logger.info("Shutting down processor...")
        self.running = False
        self.consumer.close()
        if self.mongo_client:
            self.mongo_client.close()
        sys.exit(0)
        
    def process_paper(self, paper_data: Dict[str, Any]) -> bool:
        """
        Processes a single arXiv paper by extracting metadata and text content.
        """
        try:
            # Parse PDF if not already included
            if 'text_content' not in paper_data:
                logger.info(f"Parsing PDF for paper {paper_data.get('paper_id')}")
                text_content = self.parse_pdf(paper_data['paper_id'])
                if text_content:
                    paper_data['text_content'] = text_content
                else:
                    logger.warning(f"Failed to parse PDF for {paper_data.get('paper_id')}")
                    paper_data['text_content'] = ""
            
            # Insert into database
            # Get all main topics
            collections = paper_data.get('topic', [])            
            self.mongo_client.insert(paper_data, collections)
            logger.info(f"Successfully processed paper: {paper_data.get('paper_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process paper {paper_data.get('paper_id', 'unknown')}: {e}")
            return False
    
    def run(self):
        """
        Runs the processor to listen for messages and process arXiv papers.
        """
        if not self.mongo_client:
            logger.error("Failed to connect to MongoDB. Exiting.")
            return
            
        logger.info("Starting ArXiv processor...")
        
        try:
            for message in self.consumer.consume_messages():
                if not self.running:
                    break
                    
                paper_data = message['value']
                logger.info(f"Received message for paper: {paper_data.get('paper_id', 'unknown')}")
                
                success = self.process_paper(paper_data)
                if success:
                    logger.info(f"Message processed successfully: offset {message['offset']}")
                else:
                    logger.error(f"Failed to process message: offset {message['offset']}")
                    
        except Exception as e:
            logger.error(f"Error in processor main loop: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        if self.consumer:
            self.consumer.close()
        if self.mongo_client:
            self.mongo_client.close()
        logger.info("Processor cleanup completed")

    
    def parse_pdf(self, id):
        url = f"https://arxiv.org/pdf/{id}"
        
        # Download the PDF
        try:
            start_time = time.time()
            headers = {
                "User-Agent": "Research Agent (jonathanmhtran@gmail.com)"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an error for HTTP errors
            pdf_data = response.content

            # Parse the PDF
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            text = ""
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    text += page.get_text()
                except Exception as e:
                    logger.error(f"Failed to extract text from page {page_num} of {id}: {e}")
        except Exception as e:
            logger.error(f"Failed to download PDF for {id}: {e}")
            return None
        finally:
            doc.close()
            end_time = time.time()
            exec_time = end_time - start_time
            if exec_time < 1:
                time.sleep(1 - exec_time)
            
        return text
    
class MongoDBWrapper:
    def __init__(self):
        self.client = self.connect_to_mongo()
        if self.client:
            self.db = self.client["arxiv_db"]
        else:
            self.db = None
        
    def connect_to_mongo(self):
        """
        Establishes a connection to our MongoDB instance.
        If the connection is successful, it returns the MongoDB client.
        Otherwise, it logs an error and returns None.

        Returns:
            MongoClient: The MongoDB client instance
        """
        mongo_uri = os.getenv("MONGO_URI")
        logger.info(f"Connecting to MongoDB with URI: {mongo_uri}")
        try:
            client = MongoClient(mongo_uri)
            client.admin.command('ping')  # Test the connection
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return None
        
        return client
    
    def insert(self, document_data: Dict[str, Any], collection_names: List[str]) -> None:
        """
        Inserts a document into the specified MongoDB collection(s).

        If an error occurs during insertion, it logs the error.

        Args:
            document_data (Dict[str, Any]): The data to insert.
            collection_name (str): The name of the collection to insert into.
        """
        if not self.client:
            logger.error("MongoDB client is not connected")
            return
        
        try:
            for collection_name in collection_names:
                collection = self.db[collection_name]
                result = collection.insert_one(document_data)
                logger.info(f"Inserted document with ID: {result.inserted_id} into collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
    

if __name__ == "__main__":
    processor = ArxivProcessor()
    processor.run()