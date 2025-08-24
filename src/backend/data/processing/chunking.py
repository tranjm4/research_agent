"""
File: src/backend/data/chunking/chunking.py

This file is responsible for chunking the article content into smaller, manageable pieces for processing.
"""

from message_queue.message_queue import KafkaProducerWrapper, KafkaConsumerWrapper
from pymongo.mongo_client import MongoClient
from typing import Dict, Any, List, Tuple, Iterable
from typing_extensions import TypedDict
import logging
import signal
import tiktoken

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
import os
load_dotenv()
TOPIC_NAME_CHUNKS = os.getenv('TOPIC_NAME_CHUNKS', 'chunks')
TOPIC_NAME_DOCS = os.getenv('TOPIC_NAME_DOCS', 'docs')

class Document(TypedDict):
    """
    The MongoDB Document structure
    """
    _id: str
    title: str
    authors: List[str]
    paper_id: str
    identifier: str
    datestamp: str
    created_date: str
    updated_date: str
    abstract: str
    topic: List[str]
    subtopics: List[str]
    text_content: str
    
class ChunkDocument(TypedDict):
    """
    The MongoDB Document Chunk structure
    """
    title: str
    authors: List[str]
    paper_id: str
    abstract: str
    topic: List[str]
    subtopics: List[str]
    chunk_text: str

class DocChunker:
    def __init__(self):
        # Produces to the chunks topic after chunking
        self.producer = KafkaProducerWrapper(TOPIC_NAME_CHUNKS)
        # Reads from the docs topic
        self.consumer = KafkaConsumerWrapper(TOPIC_NAME_DOCS,
                                             consumer_group='doc_processors',
                                             max_poll_interval_ms=60000,
                                             max_poll_records=50)
        self.running = True # Indicates if the chunker is running
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/GPT-4 tokenizer
        
        signal.signal(signal.SIGINT, self.sigint_sigterm_handler)
        signal.signal(signal.SIGTERM, self.sigint_sigterm_handler)
                
    def run(self):
        """
        Main loop for reading documents from Kafka queue and chunking documents
        """
        logger.info("Starting document chunking")
        try:
            while self.running:
                message_count = 0
                for message in self.consumer.consume_messages():
                    try:
                        message_count += 1
                        if message_count % 10 == 0:
                            if not self._check_consumer_connection():
                                # If the consumer is not connected, stop processing messages
                                self.running = False
                                break
                            
                        document, chunks = self.process_message(message)
                        
                        # Send all chunks and track success
                        all_chunks_sent = True
                        for chunk_doc in self.create_chunk_documents(document, chunks):
                            if not self.producer.send_message(chunk_doc, verbose=False):
                                logger.error(f"Failed to send chunk to chunks topic: offset {message.get('offset', 'unknown')}")
                                all_chunks_sent = False
                                break
                        
                        # Only commit offset if all chunks were sent successfully
                        if all_chunks_sent:
                            commit_success = self.consumer.commit_offset(message)
                            if not commit_success:
                                logger.error("Failed to commit offset - likely kicked from consumer group. Exiting.")
                                self.running = False
                                break
                            logger.info(f"Message processed successfully: offset {message.get('offset', 'unknown')}")
                        else:
                            logger.error(f"Failed to send all chunks for offset {message.get('offset', 'unknown')} - will retry on next restart")
                            # Don't commit offset so message will be reprocessed
                            break
                            
                    except Exception as e:
                        logger.error(f"Error processing message at offset {message.get('offset', 'unknown')}: {e}")
                        # Don't commit offset so message will be reprocessed
                        break
        except Exception as e:
            logger.error(f"Error in processor main loop: {e}")
        finally:
            self.cleanup()

    def process_message(self, message: Dict[str, Any]) -> Tuple[Document, List[str]]:
        """
        Process a single message from the Kafka topic consumer.
        Retrieves chunks from the input document content,
        adds it alongside the original document.

        """
        msg = message['value']
        document = msg['data']
        document_text = document.get('text_content', '')
        chunks = self.chunk_by_tokens(document_text)

        return document, chunks
    
    def chunk_by_tokens(self, text: str, max_tokens: int = 512, overlap=128) -> List[str]:
        """
        Chunk text by token count using tiktoken tokenizer.
        
        Args:
            text (str): The text to chunk
            max_tokens (int): Maximum number of tokens per chunk (default: 256)
        
        Returns:
            List[str]: List of text chunks
        """
        try:
            # Allow special tokens to be encoded as normal text
            tokens = self.tokenizer.encode(text, disallowed_special=())
            chunks = []
            
            for i in range(0, len(tokens), max_tokens - overlap):
                # if the chunk size is smaller than half the max_tokens, pad the last chunk with the remaining
                if len(tokens) - i < (max_tokens // 2):
                    chunk_tokens = tokens[i:]
                    chunks[-1] = chunks[-1] + self.tokenizer.decode(chunk_tokens)
                else:
                    chunk_tokens = tokens[i:i + max_tokens]
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    chunks.append(chunk_text)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            # Fallback to character-based chunking if tokenization fails
            chunk_size = max_tokens * 4  # Rough estimate: 1 token ≈ 4 characters
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def create_chunk_documents(self, document: Document, chunks: List[str]) -> Iterable[ChunkDocument]:
        """
        Given the original document and its text chunks, create a ChunkDocument for each chunk.
        """
        chunk_doc_fields =['paper_id', 'title', 'authors', 'abstract', 'topic', 'subtopics']
        
        for chunk_text in chunks:
            new_document = {field: document.get(field, '') for field in chunk_doc_fields}
            new_document['chunk_text'] = chunk_text
            yield new_document

    def _check_consumer_connection(self) -> bool:
        """
        Checks if the connection to the Kafka queue is still alive.
        If not, return False; the chunker should stop processing messages.
        
        Returns:
            bool: True if the consumer is still connected, False otherwise
        """
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
            
    def sigint_sigterm_handler(self, sig, frame):
        logger.info("Received shutdown signal, stopping chunker gracefully...")
        self.running = False
        
    def cleanup(self):
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        logger.info("Chunker cleanup completed")

if __name__ == '__main__':
    chunker = DocChunker()
    chunker.run()
    