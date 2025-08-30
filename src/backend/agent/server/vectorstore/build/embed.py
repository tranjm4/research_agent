"""
File: agent/server/vectorstore/build/embed.py

This file contains the Embedder class, which is responsible for embedding documents into a vector store.
It reads from the Kafka vs-docs topic and writes to the vs-embeds topic.

It processes the vs-docs topic by embedding the documents and sending them to the vs-embeds topic.
"""

import weaviate

from kafka import KafkaConsumer, KafkaProducer

from 

client = weaviate.connect_to_local()
print(client.is_ready())

client.close()

from dotenv import load_dotenv
import os
load_dotenv()

KAFKA_TOPICS_DOCS = os.getenv("KAFKA_TOPIC_DOCS", "vs-docs")
KAFKA_TOPICS_EMBEDS = os.getenv("KAFKA_TOPIC_EMBEDS", "vs-embeds")

class Embedder:
    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model
        self.consumer = KafkaConsumer(
            KAFKA_TOPICS_DOCS,
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            group_id=os.getenv("KAFKA_GROUP_ID", "vs-embedder"),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda x: x.decode("utf-8"),
            max_poll_records=10,
        )
        self.weaviate_client = weaviate.connect_to_local()

    def run(self):
        for message in self.consumer:
            document = message.value
            embedding = self.embed_document(document)
            self.send_embedding(embedding)

    def embed_document(self, document: str):
        # Implement your embedding logic here
        

    def send_embedding(self, embedding):
        producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            value_serializer=lambda x: x.encode("utf-8"),
        )
        producer.send(KAFKA_TOPICS_EMBEDS, value=embedding)
        producer.flush()