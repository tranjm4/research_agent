"""
File: src/backend/data/processing/extract_keywords.py

This file is responsible for extracting keywords from the text documents.
The extracted keywords will be stored as document metadata for querying.

Reads from the Kafka topic (default: extracting) 
and writes to the MongoDB collection (default kwe_docs)
and produces to the Kafka topic (default: chunking)
"""

from message_queue.message_queue import KafkaConsumerWrapper, KafkaProducerWrapper
from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import os
load_dotenv()
import logging
import signal
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
import signal
import spacy

from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

KAFKA_PRODUCER_TOPIC = os.getenv("TOPIC_NAME_CHUNKING", "chunking")
KAFKA_CONSUMER_TOPIC = os.getenv("TOPIC_NAME_EXTRACTING", "extracting")

DB_NAME = os.getenv("DB_NAME")
KWE_COLLECTION_NAME = os.getenv("KWE_COLLECTION")
RUNTIME_DATE = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M")


class InputDocument(TypedDict):
    url: str
    title: str
    authors: list[str]
    paper_id: str
    identifier: str
    datestamp: str
    created_date: str
    updated_date: str
    abstract: str
    topics: list[str]
    subtopics: list[str]
    obtained_date: datetime
    parsed_date: datetime
    text_content: str
    parse_success: bool
    
class OutputDocument(InputDocument):
    keywords: list[str]
    kwe_date: datetime

class KwExtractor:
    def __init__(self):
        self.consumer = KafkaConsumerWrapper(KAFKA_CONSUMER_TOPIC,
                                             consumer_group='extractor',
                                             max_poll_interval_ms=60000,
                                             max_poll_records=100)
        self.producer = KafkaProducerWrapper(KAFKA_PRODUCER_TOPIC)
        self.running = True
        signal.signal(signal.SIGINT, self.sigint_sigterm_handler)
        signal.signal(signal.SIGTERM, self.sigint_sigterm_handler)

        self.nlp = spacy.load("en_core_web_sm") # Load spaCy model
        self.client = self.connect_to_mongo()
        
        # Batch processing setup
        self.doc_batch = []
        self.processed_messages = []
        self.batch_size = 20

    def sigint_sigterm_handler(self, sig, frame):
        logging.info("Received shutdown signal, stopping extractor gracefully...")
        self.running = False
        
    def run(self):
        """
        The main function that runs the extraction part of the pipeline.
        The KwExtractor listens from the consumer topic, processes the keywords,
            and produces the result to the producer topic
        """
        logger.info("Starting keyword extraction")
        try:
            while self.running:
                num_messages = 0
                for message in self.consumer.consume_messages():
                    try:
                        processed_doc = self.process_message(message)
                        if processed_doc:
                            # Add to batch instead of immediate insert
                            self.doc_batch.append(processed_doc)
                            self.processed_messages.append(message)
                            
                            # Process batch when it reaches batch_size
                            if len(self.doc_batch) >= self.batch_size:
                                if self.flush_batch():
                                    logger.info(f"Successfully processed batch of {len(self.processed_messages)} chunks")
                                else:
                                    logger.error("Failed to process batch - will retry on restart")
                                    break
                        else:
                            logger.error(f"Failed to process chunk at offset {message.get('offset', 'unknown')} - will retry on restart")
                            break
                            
                        num_messages += 1
                        if num_messages % 10 == 0:
                            if not self._check_consumer_connection():
                                self.running = False
                                break
                                
                    except Exception as e:
                        logger.error(f"Error processing message at offset {message.get('offset', 'unknown')}: {e}")
                        break
        except Exception as e:
            logger.error(f"Error in keyword extractor main loop: {e}")
        finally:
            # Process any remaining chunks in batch
            if self.doc_batch:
                logger.info(f"Processing final batch of {len(self.doc_batch)} chunks")
                self.flush_batch()
            self.cleanup()

    def process_message(self, message: InputDocument) -> OutputDocument:
        """Process a chunk message and extract keywords"""
        try:
            doc_data = message['value']
            
            # Extract document text for keyword processing
            doc_text = doc_data.get('text_content', '')
            if not doc_text:
                logger.warning("Empty chunk_text received, skipping")
                return doc_data  # Return empty chunk for batch consistency
            
            # Extract keywords from chunk text
            keywords = self.extract(doc_text)
            
            # Add keywords to the chunk document
            doc_data['keywords'] = keywords
            doc_data['kwe_date'] = RUNTIME_DATE
            
            return doc_data
                
        except Exception as e:
            logger.error(f"Error processing chunk message: {e}")
            return None
    
    def flush_batch(self) -> bool:
        """Insert the current batch of chunks and commit their offsets."""
        if not self.doc_batch:
            return True
            
        try:
            
            # Create a copy of the docs
            mongo_docs = []
            for doc in self.doc_batch:
                clean_doc = {k:v for k,v in doc.items() if k != "_id"}
                mongo_docs.append(clean_doc)
                    
            # Batch insert all docs to MongoDB
            result = self.client[DB_NAME][KWE_COLLECTION_NAME].insert_many(mongo_docs, ordered=False)
            
            if result.inserted_ids:
                # Send all processed documents to next topic (e.g., chunking)
                docs_sent = 0
                for document in self.doc_batch:
                    if self.producer.send_message(document, verbose=False):
                        docs_sent += 1
                    else:
                        logger.warning(f"Failed to send document {document.get('paper_id', 'unknown')} to chunking topic")
                logger.info(f"Successfully send {docs_sent} documents to chunking topic")
                
                # Commit all message offsets only after successful
                # MongoDB insert AND Kafka send
                for message in self.processed_messages:
                    commit_success = self.consumer.commit_offset(message)
                    if not commit_success:
                        logger.error("Failed to commit offset - likely kicked from consumer group.")
                        return False
                
                logger.info(f"Successfully inserted {len(result.inserted_ids)} chunks with keywords")
                
                # Clear the batches
                self.doc_batch.clear()
                self.processed_messages.clear()
                return True
            else:
                logger.error("Failed to insert batch into MongoDB")
                return False
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return False

    def extract(self, text: str) -> List[str]:
        """
        Extracts keywords from a given text excerpt.
        Uses the configured NLP method to extract 50 keywords
        TODO: Improve pruning/method to reduce noisy results
        
        Args:
            text (str): The input text to extract keywords from
            
        Returns:
            List[str]: The list of keywords that the extraction method produces
        """
        try:
            doc = self.nlp(text)
            # Extract named entities and noun phrases as keywords
            keywords = []
            
            # Add named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    keywords.append(ent.text.lower().strip())
            
            # Add noun phrases
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                if len(chunk_text.split()) <= 3 and len(chunk_text) < 16:
                    keywords.append(chunk.text.lower().strip())
            
            # Remove duplicates and empty strings
            return list(set([kw for kw in keywords if kw and len(kw) > 2]))[:50]
            
        except Exception as e:
            logging.error(f"Error extracting keywords from text: {e}")
            return []
    

    def connect_to_mongo(self):
        try:
            self.client = MongoClient(os.getenv("MONGO_URI"))
            self.client.admin.command('ping')
            
            logging.info("Connected to MongoDB")
            return self.client
        except ConnectionFailure as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            
    def _check_consumer_connection(self) -> bool:
        try:
            assignments = self.consumer.consumer.assignment()
            if not assignments:
                logger.error("Not assigned to any partitions - chunker likely kicked from group. Exiting.")
                self.running = False
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking consumer group assignment: {e}. Exiting chunker.")
            self.running = False
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'consumer') and self.consumer:
            self.consumer.close()
        if hasattr(self, 'client') and self.client:
            self.client.close()
        logger.info("Keyword extractor cleanup completed")

if __name__ == '__main__':
    extractor = KwExtractor()
    extractor.run()
    # print(os.getenv("MONGO_URI"))
    # client = MongoClient(os.getenv("MONGO_URI"))
    # print(client.admin.command('ping'))
    # db = client['arxiv_db']['cs']
    # for doc in db.find():
        # print(doc)
        # break