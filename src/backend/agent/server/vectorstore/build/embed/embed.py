"""
File: agent/server/vectorstore/build/embed.py

This file contains the Embedder class, which is responsible for embedding documents into a vector store.
It reads from the Kafka vs-docs topic and writes to the vs-embeds topic.

It processes the vs-docs topic by embedding the documents and sending them to the vs-embeds topic.
"""

import json
import logging
import time
import tempfile
import uuid
from datetime import datetime

from openai import OpenAI
from kafka import KafkaConsumer
from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure

from typing_extensions import TypedDict
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_POLL_INTERVAL = int(os.getenv("BATCH_POLL_INTERVAL", "120"))  # seconds between status checks
MAX_BATCH_WAIT_TIME = int(os.getenv("MAX_BATCH_WAIT_TIME", "86400"))  # 24 hours max wait
MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "5"))

class VSMetadata(TypedDict):
    paper_id: str
    title: str
    authors: List[str]
    topics: List[str]
    subtopics: List[str]
    keywords: List[str]
    chunk_text: Optional[str]

class VSDocument(TypedDict):
    content: str
    metadata: VSMetadata

class VSEmbedding(TypedDict):
    embedding: List[float]
    metadata: VSMetadata


class Embedder:
    def __init__(self, embedding_model: str="text-embedding-3-small"):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.embedding_model = embedding_model
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        self.batch_size = int(os.getenv("BATCH_SIZE", "20000"))

        self.kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.kafka_topic_docs = os.getenv("KAFKA_TOPIC_DOCS", "vs-docs")
        self.consumer = KafkaConsumer(
            self.kafka_topic_docs,
            bootstrap_servers=self.kafka_bootstrap_servers,
            group_id=os.getenv("KAFKA_GROUP_ID", "vs-embedder"),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda x: x.decode("utf-8"),
            max_poll_records=self.batch_size,
            fetch_max_bytes=400 * 1024 * 1024, # 400MB
            fetch_max_wait_ms=15000, # 15 seconds wait for 20000 records
            max_partition_fetch_bytes = 120 * 1024 * 1024 # 120MB
        )

        try:
            MONGO_DB_URI = os.getenv("MONGO_DB_URI")
            client = MongoClient(MONGO_DB_URI)
            client.admin.command('ping')
            self.db = client["arxiv_db"]
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise ConnectionFailure("Failed to connect to MongoDB")

        logger.info("Batch Embedder initialized")
        logger.info(f"Batch size: {self.batch_size} documents, Poll interval: {BATCH_POLL_INTERVAL}s")
    

    def run(self):
        logger.info(f"Starting Batch Embedder, consuming from {self.kafka_topic_docs}")
        batch = []
        
        while True:
            message_batch = self.consumer.poll(timeout_ms=1000)
            if not message_batch:
                break

            logger.info(f"Received {len(message_batch)} messages")
            
            for messages in message_batch.values():
                for message in messages:
                    try:
                        document_data = json.loads(message.value)
                        batch.append(document_data)

                        if len(batch) >= self.batch_size:
                            logger.info(f"Processing batch with {len(batch)} documents")
                            self.process_batch_job(batch)
                            batch = []

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue
            logger.info(f"Processed batch with {len(batch)} documents")
        if batch:
            logger.info(f"Processing final batch with {len(batch)} documents")
            self.process_batch_job(batch)

        logger.info("Batch processing complete")


    def process_batch_job(self, documents: List[VSDocument]):
        """Process a batch of documents using OpenAI Batch API"""
        try:
            # Create batch file
            batch_file_id = self.create_batch_file(documents)
            
            # Submit batch job
            batch_job = self.submit_batch_job(batch_file_id)

            # Store batch job information in MongoDB
            self.db["batch_jobs"].insert_one({
                "batch_id": batch_job.id,
                "date_submitted": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "submitted"
            })

        except Exception as e:
            logger.error(f"Error processing batch job: {e}")
            raise
    
    def create_batch_file(self, documents: List[VSDocument]) -> str:
        """Create JSONL batch file"""
        batch_requests = []
        batch_metadata = [] # for storing metadata
        for i, doc in enumerate(documents):
            if not doc["content"].strip():
                continue
            custom_id = f"batch_doc_{i}_{doc['metadata']['paper_id']}"
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": self.embedding_model,
                    "input": doc["content"],
                    "encoding_format": "float"
                }
            }
            doc["metadata"]["batch_id"] = custom_id
            batch_requests.append(request)
            batch_metadata.append(doc["metadata"])
        
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for request in batch_requests:
                json.dump(request, f)
                f.write('\n')
            temp_file_path = f.name
        
        try:
            # Upload file to OpenAI
            with open(temp_file_path, 'rb') as f:
                batch_file = self.openai_client.files.create(
                    file=f,
                    purpose='batch'
                )
            logger.info(f"Uploaded batch file {batch_file.id}")
            # Upload metadata to MongoDB
            self.db["embeddings_metadata"].insert_many(batch_metadata)
            return batch_file.id
        finally:
            os.unlink(temp_file_path)
    
    def submit_batch_job(self, batch_file_id: str):
        """Submit batch job to OpenAI"""
        batch_job = self.openai_client.batches.create(
            input_file_id=batch_file_id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )
        logger.info(f"Submitted batch job {batch_job.id}")
        return batch_job
    
                
if __name__ == "__main__":
    embedder = Embedder()
    embedder.run()