"""
File: server/vectorstore/build/loader.py

This file contains the DBLoader class, which is responsible for loading the MongoDB documents into the Kafka topic.
"""
from kafka import KafkaProducer
from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure

from tqdm import tqdm
from time import sleep

from typing import Dict, Any, Tuple
from typing_extensions import TypedDict

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import os
load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_TOPIC_DOCS = os.getenv("KAFKA_TOPIC_DOCS")
MONGO_DB_URI = os.getenv("MONGO_DB_URI")

class QueueMessage(TypedDict):
    content: str
    metadata: Dict[str, Any]

class DBLoader:
    def __init__(self, db_uri):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
            batch_size=16384,
            linger_ms=100,
            buffer_memory=67108864,  # 64MB buffer
            max_block_ms=5000,
            retries=3,
            acks=1
        )
        self.uri = db_uri
        self.db = self.connect_to_db(db_uri)
        
        
    def connect_to_db(self, uri):
        client = MongoClient(uri)
        try:
            client.admin.command("ping")
            print("Connected to MongoDB")
        except ConnectionFailure:
            print("Failed to connect to MongoDB")
        
        return client[os.getenv("MONGO_DB_NAME")]

    def send_message_batch(self, messages: list[Tuple[Dict[str, Any], int]], progress: Tuple[int,int]) -> None:
        """
        Sends a list of messages in batches to the Kafka topic.
        
        Args:
            message (Dict[str, Any]): The message to send.
        """
        try:
            logger.info(f"Sending batch {progress[0]:0>6}/{progress[1]} with {len(messages)} messages to Kafka...")
            for message in messages:
                queue_document = self.construct_queue_document(message)
                self.producer.send(KAFKA_TOPIC_DOCS, value=str(queue_document).encode('utf-8'))
        except Exception as e:
            logger.error(f"Error sending messages to Kafka: {e}")
            
    def construct_queue_document(self, doc: Dict[str, Any]) -> QueueMessage:
        """
        Given the MongoDB document, constructs a QueueMessage
        """
        return QueueMessage(
            content=doc.get("chunk_text", "").strip(),
            metadata={
                "paper_id": doc.get("paper_id", ""),
                "title": doc.get("title", ""),
                "authors": doc.get("authors", []),
                "topics": doc.get("topic", []),
                "subtopics": doc.get("subtopic", []),
                "keywords": doc.get("keywords", [])
            }
        )

    def load_documents(self, collection_name: str) -> None:
        """
        Loads documents from a MongoDB collection and sends them to a Kafka topic.
        """
        collection = self.db[collection_name]
        batch = []
        logger.info("Starting to load documents into Kafka...")
        total_batches = collection.count_documents({}) // 1000 + 1
        num_batches = 1
        for doc in collection.find():
            batch.append(doc)
            if len(batch) >= 1000:  # Send batch of 100 messages
                self.send_message_batch(batch, (num_batches, total_batches))
                batch = []
                num_batches += 1
        if batch:
            self.send_message_batch(batch, (num_batches, total_batches))

        logger.info("Finished loading documents into Kafka.")
        
if __name__ == "__main__":
    db_loader = DBLoader(db_uri=MONGO_DB_URI)
    db_loader.load_documents(collection_name="arxiv_chunks")