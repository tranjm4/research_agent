"""
File: src/backend/data/kafka/message_queue.py
"""

from kafka import KafkaConsumer, KafkaProducer, TopicPartition, OffsetAndMetadata
from kafka.errors import KafkaError
from typing import Dict, Any, Optional, List
import json
import hashlib
import threading

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv
load_dotenv()
TOPIC_NAME = os.getenv('TOPIC_NAME', 'arxiv_papers')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

class KafkaProducerWrapper:
    def __init__(self, topic_name: str):
        self._message_counter = 0
        self._counter_lock = threading.Lock()
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            retries=3,
            acks='all',
            partitioner=self._custom_partitioner,
            max_request_size=104857600,  # 100MB
            buffer_memory=134217728      # 128MB
        )
        self.topic_name = topic_name
        
    def _custom_partitioner(self, key_bytes, all_partitions, available_partitions):
        """Custom partitioner to distribute messages evenly across partitions"""
        if key_bytes is None:
            # Use thread-safe round-robin for messages without keys
            with self._counter_lock:
                partition = self._message_counter % len(all_partitions)
                self._message_counter += 1
                return partition
        
        # Hash the key and distribute across all partitions
        return int(hashlib.md5(key_bytes).hexdigest(), 16) % len(all_partitions)
        
    def send_message(self, message: Dict[str, Any], key: Optional[str] = None, verbose=True) -> bool:
        try:
            # Check message size before sending
            message_bytes = json.dumps(message).encode('utf-8')
            message_size = len(message_bytes)
            
            if message_size > 50 * 1024 * 1024:  # 50MB limit
                logger.warning(f"Message too large ({message_size/1024/1024:.1f}MB), skipping document {key or 'unknown'}")
                return False
            
            future = self.producer.send(self.topic_name, value=message, key=key)
            record_metadata = future.get(timeout=10)
            if verbose:
                logger.info(f"Message sent successfully to {record_metadata.topic} partition {record_metadata.partition}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            return False
            
    def close(self):
        self.producer.close()

# Consumer wrapper class for listening for messages to be processed
class KafkaConsumerWrapper:
    def __init__(self, topic: str, consumer_group: str = 'arxiv_processors',
                 session_timeout_ms=30000, heartbeat_interval_ms=10000,
                 max_poll_interval_ms=180000, max_poll_records=5):
        self.topic = topic
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
            group_id=consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            session_timeout_ms=session_timeout_ms,
            heartbeat_interval_ms=heartbeat_interval_ms,
            max_poll_interval_ms=max_poll_interval_ms,
            max_poll_records=max_poll_records
        )
        
    def consume_messages(self):
        try:
            while True:
                # Manual poll with timeout
                msg_pack = self.consumer.poll(timeout_ms=1000)
                if not msg_pack:
                    continue
                    
                for tp, messages in msg_pack.items():
                    for message in messages:
                        yield {
                            'key': message.key,
                            'value': message.value,
                            'partition': message.partition,
                            'offset': message.offset
                        }
        except KeyboardInterrupt:
            logger.info("Consumer interrupted. Shutting down...")
        finally:
            self.consumer.close()
            
    def commit_offset(self, message_dict):
        """Manually commit the offset for a successfully processed message"""
        try:
            # Seek to the next position and then commit
            tp = TopicPartition(self.topic, message_dict['partition'])
            self.consumer.seek(tp, message_dict['offset'] + 1)
            self.consumer.commit()
            logger.info(f"Committed offset {message_dict['offset'] + 1} for partition {message_dict['partition']}")
            return True
        except Exception as e:
            logger.error(f"Failed to commit offset: {e}")
            return False
            
    def close(self):
        self.consumer.close()
