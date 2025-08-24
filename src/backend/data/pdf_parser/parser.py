"""
PROCESSOR

Consumes messages from Kafka queue and processes arXiv paper data.
Handles PDF parsing, text extraction, and database insertion.
"""

import logging
from typing import Dict, Any, List, Optional
from kafka import KafkaConsumer
import signal
import fitz # PyMuPDF
from pymongo import ReplaceOne
import requests
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure

from message_queue.message_queue import KafkaConsumerWrapper, KafkaProducerWrapper

from dotenv import load_dotenv
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOPIC_NAME_PAPERS = os.getenv('TOPIC_NAME_PAPERS', 'arxiv_papers')
TOPIC_NAME_DOCS = os.getenv('TOPIC_NAME_DOCS', 'arxiv_docs')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

class ArxivParser:
    def __init__(self):
        self.consumer = KafkaConsumerWrapper(TOPIC_NAME_PAPERS)
        self.producer = KafkaProducerWrapper(TOPIC_NAME_DOCS)
        self.mongo_client = MongoDBWrapper()
        self.running = True
        
        # Batch processing setup
        self.document_batch = []
        self.processed_messages = []
        self.batch_size = 20
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        logger.info("Received shutdown signal, stopping parser gracefully...")
        self.running = False
        
    def process_paper(self, paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a single arXiv paper by extracting metadata and text content.
        Returns the processed document for batch insertion.

        Returns:
            Optional[Dict[str, Any]]: Processed document data or None if failed.
        """
        try:
            paper_id = paper_data.get('paper_id')
            topics = paper_data.get('topic', [])
            
            if not topics:
                logger.warning(f"No topics found for paper {paper_id}, skipping")
                return None
            else:
                collection_name = topics[0]
                # Check if paper already exists in MongoDB
                if self.mongo_client.paper_exists(paper_id, collection_name):
                    logger.info(f"Paper {paper_id} already exists in database, skipping")
                    return paper_data  # Return existing paper for batch consistency
            
            # Parse PDF if not already included
            if 'text_content' not in paper_data:
                text_content = self.parse_pdf(paper_id)
                if text_content:
                    paper_data['text_content'] = text_content
                else:
                    logger.warning(f"Failed to parse PDF for {paper_id}")
                    paper_data['text_content'] = ""
            
            return paper_data
            
        except Exception as e:
            logger.error(f"Failed to process paper {paper_data.get('paper_id', 'unknown')}: {e}")
            return None
    
    def flush_batch(self) -> bool:
        """
        Insert the current batch of documents and commit their offsets.
        
        Returns:
            bool: True if batch processing was successful, False otherwise.
        """
        if not self.document_batch:
            return True
            
        try:
            # Insert all documents into all_docs collection
            success = self.mongo_client.batch_insert(self.document_batch, collection_name="all_docs")

            if success:
                # Send all processed documents to 'docs' topic
                docs_sent = 0
                for document in self.document_batch:
                    doc_message = {
                        "collection": "all_docs",  # or use document.get('topic', ['unknown'])[0]
                        "data": document
                    }
                    
                    if self.producer.send_message(doc_message, verbose=False):
                        docs_sent += 1
                    else:
                        logger.warning(f"Failed to send document {document.get('paper_id', 'unknown')} to docs topic - likely too large, skipping")
                        # Don't return False here - continue with other documents
                
                logger.info(f"Successfully sent {docs_sent} documents to docs topic")
                
                # Commit all message offsets only after successful MongoDB insert AND Kafka send
                for message in self.processed_messages:
                    commit_success = self.consumer.commit_offset(message)
                    if not commit_success:
                        logger.error("Failed to commit offset - likely kicked from consumer group.")
                        return False
                
                # Clear the batches
                self.document_batch.clear()
                self.processed_messages.clear()
                return True
            else:
                logger.error("Failed to insert batch into MongoDB")
                return False
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
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
            message_count = 0
            for message in self.consumer.consume_messages():
                if not self.running:
                    break
                
                message_count += 1
                
                # Check consumer group connection every 10 messages
                if message_count % 10 == 0:
                    try:
                        assignments = self.consumer.consumer.assignment()
                        if not assignments:
                            logger.error("Not assigned to any partitions - processor kicked from group. Exiting.")
                            self.running = False
                            break
                    except Exception as e:
                        logger.error(f"Error checking consumer group assignment: {e}. Exiting processor.")
                        self.running = False
                        break
                    
                paper_data = message['value']
                logger.info(f"Received message for paper: {paper_data.get('paper_id', 'unknown')}")
                
                processed_doc = self.process_paper(paper_data)
                if processed_doc:
                    # Add to batch instead of immediate insert
                    self.document_batch.append(processed_doc)
                    self.processed_messages.append(message)
                    
                    # Process batch when it reaches batch_size
                    if len(self.document_batch) >= self.batch_size:
                        if self.flush_batch():
                            logger.info(f"Successfully processed batch of {len(self.processed_messages)} documents")
                        else:
                            logger.error("Failed to process batch - will retry on restart")
                            break
                else:
                    logger.error(f"Failed to process message: offset {message['offset']} - will retry on restart")
                    break
                    
        except Exception as e:
            logger.error(f"Error in processor main loop: {e}")
        finally:
            # Process any remaining documents in batch
            if self.document_batch:
                logger.info(f"Processing final batch of {len(self.document_batch)} documents")
                self.flush_batch()
            self.cleanup()
            
    def cleanup(self):
        if self.consumer:
            self.consumer.close()
        if self.mongo_client:
            self.mongo_client.close()
        logger.info("Processor cleanup completed")

    
    def parse_pdf(self, id):
        url = f"https://arxiv.org/pdf/{id}"
        doc = None
        
        # Download the PDF
        try:
            start_time = time.time()
            headers = {
                "User-Agent": "Research Agent (jonathanmhtran@gmail.com)"
            }
            response = requests.get(url, headers=headers, timeout=20)
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
            if doc:
                doc.close()
            end_time = time.time()
            exec_time = end_time - start_time
            if exec_time < 0.5:
                time.sleep(0.5 - exec_time)
            
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
    
    def paper_exists(self, paper_id: str, collection_name: str) -> bool:
        """
        Checks if a paper with the given paper_id already exists in MongoDB.
        Checks the first available collection for efficiency.
        
        Args:
            paper_id (str): The paper ID to check for.
            collection_name (str): The collection name to search for the paper ID.
            
        Returns:
            bool: True if the paper exists, False otherwise.
        """
        if not self.client:
            logger.error("MongoDB client is not connected")
            return False
            
        try:
            collection = self.db[collection_name]
            return bool(collection.find_one({"paper_id": paper_id}))
        except Exception as e:
            logger.error(f"Failed to check if paper exists: {e}")
            return False

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
                result = collection.replace_one({"paper_id": document_data["paper_id"]}, document_data, upsert=True)
                logger.info(f"Inserted document with ID: {result.upserted_id} into collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")

    def batch_insert(self, documents: List[Dict[str, Any]], collection_name: str) -> bool:
        """
        Batch insert documents into a single collection.

        Args:
            documents: List of documents to insert.
            collection_name: The name of the collection to insert into.

        Returns:
            bool: True if the batch insert succeeded, False otherwise.
        """
        if not self.client:
            logger.error("MongoDB client is not connected")
            return False

        try:
            if not documents:
                return True
                
            collection = self.db[collection_name]
            operations = [
                ReplaceOne({"paper_id": doc["paper_id"]}, doc, upsert=True)
                for doc in documents if "paper_id" in doc
            ]
            result = collection.bulk_write(operations, ordered=False)
            logger.info(f"Batch inserted {len(documents)} documents into {collection_name}: "
                        f"{result.upserted_count} new, {result.modified_count} updated")
            return True

        except Exception as e:
            logger.error(f"Failed to batch insert documents: {e}")
            return False
            

    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    

if __name__ == "__main__":
    parser = ArxivParser()
    parser.run()