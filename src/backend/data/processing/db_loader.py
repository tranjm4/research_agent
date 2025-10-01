"""
File: src/backend/data/chunking/db_loader.py

This file is responsible for loading the MongoDB documents into the Kafka topic.
The files will be processed downstream
"""

from message_queue.message_queue import KafkaProducerWrapper
from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

class DocLoader:
    def __init__(self, topic_name: str = None):
        if topic_name is None:
            topic_name = os.getenv('TOPIC_NAME_DOCS', 'docs')
        self.topic_name = topic_name
        self.producer = KafkaProducerWrapper(topic_name)
        self.db = self.connect_to_mongo()
        if self.db is None:
            logger.error("Failed to connect to MongoDB")
            raise ConnectionFailure("Could not connect to MongoDB")

    def load_documents_to_kafka(self):
        if self.db is None:
            logger.error("DB was never initialized. Skipping document loading.")
            return

        collections = self.db.list_collection_names()
        for collection_name in collections:
            self.load_collection_to_kafka(collection_name)

    def load_collection_to_kafka(self, collection_name: str):
        """
        Given a collection name, load all documents to the Kafka topic 'docs'.
        Each document should be provided in the format:
            {
                "collection": "<collection_name>",
                "data": <document data>
            }
        """
        collection = self.db[collection_name]
        i = 0
        logger.info(f"{'-' * 50}\nLoading documents from {collection_name} to Kafka\n{'-' * 50}")
        
        try:
            for document in collection.find():
                try:
                    # Remove _id from document
                    clean_document = {k:v for k,v in document.items() if k != "_id"}
                    
                    self.producer.send_message(clean_document)
                    i += 1
                    
                    if i % 500 == 0:
                        logger.info(f"Sent {i} documents from {collection_name} to Kafka")
                        
                except Exception as e:
                    logger.error(f"Error processing document {document.get('_id', 'unknown')}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error loading collection {collection_name}: {e}")
        
        logger.info(f"Finished loading {i} documents from {collection_name}")

    def deduplicate_documents(self) -> int:
        """
        Removes duplicate documents across all collections
        
        Returns the number of duplicates removed
        """
        if self.db is None:
            logger.error("DB was never initialized. Skipping document deduplication.")
            return 0

        # Implement deduplication logic with batching for memory safety
        seen = set()
        num_duplicates = 0
        batch_size = 1000
        
        for collection_name in self.db.list_collection_names():
            collection = self.db[collection_name]
            skip = 0
            
            while True:
                documents = list(collection.find().skip(skip).limit(batch_size))
                if not documents:
                    break
                    
                for document in documents:
                    doc_id = document.get("_id")
                    if doc_id in seen:
                        collection.delete_one({"_id": doc_id})
                        num_duplicates += 1
                    else:
                        seen.add(doc_id)
                        
                skip += batch_size
                logger.info(f"Processed {skip} documents from {collection_name}")
                
        return num_duplicates

    def connect_to_mongo(self):
        """
        Connect to the MongoDB database. Returns None if unsuccessful
        """
        load_dotenv()
        client = MongoClient(os.getenv("MONGO_URI"))
        try:
            client.admin.command('ping')
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            return None
        
        db = client[os.getenv("DB_NAME")]
        return db

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load MongoDB documents to Kafka')
    parser.add_argument('--collection', type=str, required=True, help='Collection to load')
    parser.add_argument('--topic', type=str, required=True, help='Kafka topic name')
    args = parser.parse_args()
    
    logger.info(f"Loading documents from {args.collection} to {args.topic} Kafka topic")

    doc_loader = DocLoader(topic_name=args.topic)
    doc_loader.load_collection_to_kafka(args.collection)