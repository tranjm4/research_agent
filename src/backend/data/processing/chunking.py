"""
File: src/backend/data/chunking/chunking.py

This file is responsible for chunking the article content into smaller, manageable pieces for processing.
"""

from message_queue.message_queue import KafkaProducerWrapper, KafkaConsumerWrapper
from pymongo.mongo_client import MongoClient
from pymongo.errors import PyMongoError
from typing import Dict, Any, List, Tuple, Iterable, Literal
from typing_extensions import TypedDict
import logging
import signal
import tiktoken
import nltk
nltk.download("punkt_tab")

from argparse import ArgumentParser

from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
import os
load_dotenv()
TOPIC_NAME_CONSUMER = os.getenv('TOPIC_NAME_CHUNKING', 'chunking')
RUNTIME_DATE = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

if not DB_NAME or not MONGO_URI:
    raise Exception("Failed to get DB_NAME or MONGO_URI from environment variables")

class InputDocument(TypedDict):
    """
    The MongoDB Document structure
    """
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
    keywords: List[str]
    kwe_date: datetime
    
class OutputDocument(InputDocument):
    chunked_date: datetime
    chunk_text: str
    
ChunkingStrategy = Literal["token", "semantic", "sentence"]

class ChunkingStrategyInterface:
    """Base class for chunking strategies"""
    def chunk(self, text: str) -> List[str]:
        raise NotImplementedError

class TokenChunker(ChunkingStrategyInterface):
    def __init__(self, max_tokens: int = 256, overlap: int = 64):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> List[str]:
        try:
            tokens = self.tokenizer.encode(text, disallowed_special=())
            chunks = []

            for i in range(0, len(tokens), self.max_tokens - self.overlap):
                if len(tokens) - i < (self.max_tokens // 2) and chunks:
                    chunk_tokens = tokens[i:]
                    chunks[-1] = chunks[-1] + self.tokenizer.decode(chunk_tokens)
                else:
                    chunk_tokens = tokens[i:i + self.max_tokens]
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    chunks.append(chunk_text)

            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            # Fallback to character-based chunking
            chunk_size = self.max_tokens * 4
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

class SemanticChunker(ChunkingStrategyInterface):
    """Chunks text by paragraphs and sections, preserving semantic boundaries"""
    def __init__(self, max_chars: int = 1024, overlap: int = 128):
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        try:
            # Split by paragraphs (double newlines)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

            chunks = []
            current_chunk = ""

            for para in paragraphs:
                # If adding this paragraph exceeds max, save current chunk
                if len(current_chunk) + len(para) > self.max_chars and current_chunk:
                    chunks.append(current_chunk)
                    # Start new chunk with overlap from previous
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk += ("\n\n" if current_chunk else "") + para

            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk)

            return chunks if chunks else [text]
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            # Fallback to simple character chunking
            return [text[i:i + self.max_chars] for i in range(0, len(text), self.max_chars)]

class SentenceChunker(ChunkingStrategyInterface):
    """Chunks text by sentences, grouping until token limit"""
    def __init__(self, max_tokens: int = 256, overlap_sentences: int = 2):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def chunk(self, text: str) -> List[str]:
        try:
            # Tokenize into sentences
            sentences = nltk.sent_tokenize(text)

            chunks = []
            current_chunk = []
            current_tokens = 0

            for sentence in sentences:
                sentence_tokens = len(self.tokenizer.encode(sentence))

                # If adding this sentence exceeds max, save current chunk
                if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Start new chunk with overlap sentences
                    current_chunk = current_chunk[-self.overlap_sentences:] if len(current_chunk) > self.overlap_sentences else []
                    current_tokens = sum(len(self.tokenizer.encode(s)) for s in current_chunk)

                current_chunk.append(sentence)
                current_tokens += sentence_tokens

            # Add final chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks if chunks else [text]
        except Exception as e:
            logger.error(f"Error in sentence chunking: {e}")
            # Fallback to character chunking
            chunk_size = self.max_tokens * 4
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

class DocChunker:
    def __init__(self, consumer_group_name: str, collection_name: str, chunking_strategy: ChunkingStrategyInterface):
        # Reads from the docs topic
        self.consumer_group_name = consumer_group_name
        self.consumer = KafkaConsumerWrapper(TOPIC_NAME_CONSUMER,
                                             consumer_group=consumer_group_name,
                                             max_poll_interval_ms=60000,
                                             max_poll_records=50)
        self.running = True # Indicates if the chunker is running
        self.chunking_strategy = chunking_strategy

        self.client = MongoClient(MONGO_URI)
        try:
            self.client.admin.command("ping")
        except Exception:
            raise PyMongoError("Failed to connect to MongoDB instance")

        self.collection_name = collection_name

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
                    if not self.running:
                        break
                    try:
                        message_count += 1
                        if message_count % 10 == 0:
                            if not self._check_consumer_connection():
                                # If the consumer is not connected, stop processing messages
                                self.running = False
                                break
                            
                        document, chunks = self.process_message(message)
                        
                        # Send all chunks and track success
                        batch_size = 32
                        batch = []
                        for chunk_doc in self.create_chunk_documents(document, chunks):
                            batch.append(chunk_doc)
                            if len(batch) == batch_size:
                                if not self._batch_insert_docs_mongo(batch):
                                    logger.error("Failed to insert batch - will rety on restart")
                                    break
                                batch.clear()
                        # Last batch (if any remaining)
                        if batch:
                            if not self._batch_insert_docs_mongo(batch):
                                logger.error("Failed to insert batch - will rety on restart")
                                break
                        
                        # Only commit offset if all chunks were sent successfully
                        commit_success = self.consumer.commit_offset(message)
                        if not commit_success:
                            logger.error("Failed to commit offset - likely kicked from consumer group. Exiting.")
                            self.running = False
                            break
                        
                        logger.info(f"Message processed successfully: offset {message.get('offset', 'unknown')}")
                            
                    except Exception as e:
                        logger.error(f"Error processing message at offset {message.get('offset', 'unknown')}: {e}")
                        # Don't commit offset so message will be reprocessed
                        break
        except Exception as e:
            logger.error(f"Error in processor main loop: {e}")
        finally:
            self.cleanup()

    def process_message(self, message: Dict[str, Any]) -> Tuple[InputDocument, List[str]]:
        """
        Process a single message from the Kafka topic consumer.
        Retrieves chunks from the input document abstract,
        adds it alongside the original document.
        """
        document = message['value']

        # Use abstract for chunking since full text is no longer preprocessed
        abstract = document.get('abstract', '')
        title = document.get('title', '')

        # Combine title and abstract for chunking
        document_text = f"{title}\n\n{abstract}" if title and abstract else (title or abstract)

        if not document_text:
            logger.warning(f"No text available for chunking paper {document.get('paper_id', 'unknown')}")
            chunks = []
        else:
            chunks = self.chunking_strategy.chunk(document_text)

        return document, chunks

    def create_chunk_documents(self, document: InputDocument, chunks: List[str]) -> Iterable[OutputDocument]:
        """
        Given the original document and its text chunks, create a ChunkDocument for each chunk.
        """
        for chunk_text in chunks:
            # make a copy of the original document
            new_document = {k:v for k,v in document.items()}
            new_document['chunk_text'] = chunk_text
            new_document['chunked_date'] = RUNTIME_DATE
            yield new_document
            
    def _batch_insert_docs_mongo(self, batch: List[OutputDocument]) -> None:
        """
        Performs a batch insert (insert_many) into the MongoDB instance
        
        Args:
            batch (List[OutputDocument]): The batch of resulting documents to insert
        """
        try:
            db = self.client[DB_NAME]
            collection = db[self.collection_name]
            clean_batch = [{k: v for k, v in doc.items() if k != "_id"} for doc in batch]
            result = collection.insert_many(clean_batch, ordered=False)
            logger.info(f"Inserted {len(result.inserted_ids)} chunks into MongoDB instance")
            return True
        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            return False

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
        logger.info("Chunker cleanup completed")

if __name__ == '__main__':
    parser = ArgumentParser(description="The script runs a specified chunking \
                                         strategy on the Kafka chunking topic")
    parser.add_argument("--consumer-group", required=True,
                        help="A unique identifier for the consumer group")
    parser.add_argument("--collection-name", required=True,
                        help="A unique identifier for the MongoDB \
                              collection for the chunked documents to be stored")
    parser.add_argument("--chunking-strategy", required=True,
                        choices=["token", "semantic", "sentence"],
                        help="The chunking strategy to use: [token | semantic | sentence]")
    args = parser.parse_args()

    # Factory pattern to create the appropriate chunking strategy
    strategy_map = {
        "token": TokenChunker(),
        "semantic": SemanticChunker(),
        "sentence": SentenceChunker()
    }
    chunking_strategy = strategy_map[args.chunking_strategy]

    chunker = DocChunker(
        consumer_group_name=args.consumer_group,
        collection_name=args.collection_name,
        chunking_strategy=chunking_strategy
    )
    chunker.run()
