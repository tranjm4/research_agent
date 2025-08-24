"""
File: src/backend/data/processing/extract_keywords.py

This file is responsible for extracting keywords from the text documents.
The extracted keywords will be stored as document metadata for querying.

Reads from the Kafka topic (chunks) and writes to the MongoDB collection (arxiv_chunks)
"""

from message_queue.message_queue import KafkaConsumerWrapper
from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import os
load_dotenv()
import logging
import signal
from typing import Dict, Any, List, Optional
import signal
import spacy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class KwExtractor:
    def __init__(self):
        self.consumer = KafkaConsumerWrapper('chunks',
                                             consumer_group='chunk_processors',
                                             max_poll_interval_ms=60000,
                                             max_poll_records=100)
        self.running = True
        signal.signal(signal.SIGINT, self.sigint_sigterm_handler)
        signal.signal(signal.SIGTERM, self.sigint_sigterm_handler)

        self.nlp = spacy.load("en_core_web_sm") # Load spaCy model
        self.db = self.connect_to_mongo()
        
        # Batch processing setup
        self.chunk_batch = []
        self.processed_messages = []
        self.batch_size = 20

    def sigint_sigterm_handler(self, sig, frame):
        logging.info("Received shutdown signal, stopping extractor gracefully...")
        self.running = False
        
    def run(self):
        logger.info("Starting keyword extraction")
        try:
            while self.running:
                num_messages = 0
                for message in self.consumer.consume_messages():
                    try:
                        processed_chunk = self.process_message(message)
                        if processed_chunk:
                            # Add to batch instead of immediate insert
                            self.chunk_batch.append(processed_chunk)
                            self.processed_messages.append(message)
                            
                            # Process batch when it reaches batch_size
                            if len(self.chunk_batch) >= self.batch_size:
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
            if self.chunk_batch:
                logger.info(f"Processing final batch of {len(self.chunk_batch)} chunks")
                self.flush_batch()
            self.cleanup()

    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a chunk message and extract keywords"""
        try:
            chunk_data = message['value']
            
            # Extract chunk text for keyword processing
            chunk_text = chunk_data.get('chunk_text', '')
            if not chunk_text:
                logger.warning("Empty chunk_text received, skipping")
                return chunk_data  # Return empty chunk for batch consistency
            
            # Extract keywords from chunk text
            keywords = self.extract(chunk_text)
            
            # Add keywords to the chunk document
            chunk_data['keywords'] = keywords
            
            return chunk_data
                
        except Exception as e:
            logger.error(f"Error processing chunk message: {e}")
            return None
    
    def flush_batch(self) -> bool:
        """Insert the current batch of chunks and commit their offsets."""
        if not self.chunk_batch:
            return True
            
        try:
            # Batch insert all chunks
            result = self.db['arxiv_chunks'].insert_many(self.chunk_batch, ordered=False)
            
            if result.inserted_ids:
                # Commit all message offsets
                for message in self.processed_messages:
                    commit_success = self.consumer.commit_offset(message)
                    if not commit_success:
                        logger.error("Failed to commit offset - likely kicked from consumer group.")
                        return False
                
                logger.info(f"Successfully inserted {len(result.inserted_ids)} chunks with keywords")
                
                # Clear the batches
                self.chunk_batch.clear()
                self.processed_messages.clear()
                return True
            else:
                logger.error("Failed to insert batch into MongoDB")
                return False
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return False

    def extract(self, text: str) -> List[str]:
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
                if len(chunk.text.split()) <= 3:  # Limit to 3-word phrases
                    keywords.append(chunk.text.lower().strip())
            
            # Remove duplicates and empty strings
            return list(set([kw for kw in keywords if kw and len(kw) > 2]))
            
        except Exception as e:
            logging.error(f"Error extracting keywords from text: {e}")
            return []
    

    def connect_to_mongo(self):
        try:
            self.client = MongoClient(os.getenv("MONGO_URI"))
            self.client.admin.command('ping')
            
            logging.info("Connected to MongoDB")
            return self.client['arxiv_db']
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