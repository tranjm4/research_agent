"""
File: src/backend/data/kafka/message_queue.py
"""

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from typing import Dict, Any, Optional
import json

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv
load_dotenv()
TOPIC_NAME = os.getenv('TOPIC_NAME', 'arxiv_papers')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

class KafkaProducerWrapper:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            retries=3,
            acks='all'
        )
        
    def send_message(self, message: Dict[str, Any], key: Optional[str] = None):
        try:
            future = self.producer.send(TOPIC_NAME, value=message, key=key)
            record_metadata = future.get(timeout=10)
            logger.info(f"Message sent successfully to {record_metadata.topic} partition {record_metadata.partition}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            return False
            
    def close(self):
        self.producer.close()

# Consumer wrapper class for listening for messages to be processed
class KafkaConsumerWrapper:
    def __init__(self, consumer_group: str = 'arxiv_processors'):
        self.consumer = KafkaConsumer(
            TOPIC_NAME,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
            group_id=consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        
    def consume_messages(self):
        try:
            for message in self.consumer:
                yield {
                    'key': message.key,
                    'value': message.value,
                    'partition': message.partition,
                    'offset': message.offset
                }
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        finally:
            self.consumer.close()
            
    def close(self):
        self.consumer.close()
