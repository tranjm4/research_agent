"""
File: src/backend/data/vectorstore/build_vectorstore.py

This file reads documents with keywords from MongoDB and builds a FAISS vectorstore for retrieval using abstract text.
"""

from pymongo.mongo_client import MongoClient
from pymongo.errors import PyMongoError
from gridfs import GridFS
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
import os
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import sys
import io
from datetime import datetime

# Embedding imports
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

# FAISS import
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss not installed. Install with: pip install faiss-cpu")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

if not DB_NAME or not MONGO_URI:
    raise Exception("Failed to get DB_NAME or MONGO_URI from environment variables")


class FAISSVectorStoreBuilder:
    """Builds a FAISS vector store from MongoDB documents with keywords"""

    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vectorstore_path: str = "./vectorstore_data",
        batch_size: int = 100,
        index_type: str = "flat",
        storage_type: str = "file"
    ):
        """
        Initialize the FAISS VectorStoreBuilder

        Args:
            collection_name: MongoDB collection name containing documents with keywords
            embedding_model: Name of the sentence-transformers model to use
            vectorstore_path: Path to save the vectorstore
            batch_size: Batch size for processing documents
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            storage_type: Where to store the vectorstore ('file' or 'mongodb')
        """
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vectorstore_path = vectorstore_path
        self.index_type = index_type.lower()
        self.storage_type = storage_type.lower()
        self.embedding_model_name = embedding_model

        # Connect to MongoDB
        self.client = MongoClient(MONGO_URI)
        try:
            self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            raise PyMongoError(f"Failed to connect to MongoDB instance: {e}")

        self.db = self.client[DB_NAME]
        self.collection = self.db[collection_name]

        # Initialize GridFS for MongoDB storage
        if self.storage_type == "mongodb":
            self.fs = GridFS(self.db)

        # Initialize embedding model
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        if not HAS_FAISS:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")

        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Initialize FAISS index (will be created after we know the data size)
        self.faiss_index = None
        self.faiss_metadata = []

    def _init_faiss_index(self, num_vectors: int):
        """Initialize FAISS index based on the type and data size"""
        if self.index_type == "flat":
            # Flat index for exact search (good for smaller datasets)
            logger.info(f"Initializing FAISS Flat index with dimension {self.embedding_dim}")
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        elif self.index_type == "ivf":
            # IVF index for faster approximate search (good for larger datasets)
            n_clusters = min(int(np.sqrt(num_vectors)), 100)  # Heuristic for number of clusters
            logger.info(f"Initializing FAISS IVF index with {n_clusters} clusters")
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters, faiss.METRIC_INNER_PRODUCT)

        elif self.index_type == "hnsw":
            # HNSW index for fast approximate search
            logger.info(f"Initializing FAISS HNSW index")
            self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)

        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def build_vectorstore(self, limit: Optional[int] = None):
        """
        Build the vectorstore from MongoDB documents

        Args:
            limit: Optional limit on number of documents to process
        """
        # Count total documents
        total_docs = self.collection.count_documents({})
        if limit:
            total_docs = min(total_docs, limit)

        logger.info(f"Processing {total_docs} documents from collection '{self.collection_name}'")

        # Initialize FAISS index now that we know the size
        self._init_faiss_index(total_docs)
        logger.info("FAISS index initialized successfully")

        # Collect all embeddings first if using IVF (needs training)
        if self.index_type == "ivf":
            logger.info("Starting IVF index build with training...")
            self._build_with_training(total_docs, limit)
        else:
            logger.info("Starting build without training...")
            self._build_without_training(total_docs, limit)

        # Save vectorstore
        self._save_vectorstore()

    def _build_without_training(self, total_docs: int, limit: Optional[int]):
        """Build vectorstore for indexes that don't need training"""
        logger.info("Creating MongoDB cursor...")
        batch = []
        processed = 0

        cursor = self.collection.find().limit(limit) if limit else self.collection.find()
        logger.info("Cursor created, starting iteration...")

        # Disable tqdm if not in a TTY (e.g., Docker)
        use_tqdm = sys.stderr.isatty()

        with tqdm(total=total_docs, desc="Building vectorstore", disable=not use_tqdm) as pbar:
            for doc in cursor:
                batch.append(doc)

                if len(batch) >= self.batch_size:
                    self._process_batch(batch)
                    processed += len(batch)
                    pbar.update(len(batch))

                    # Log progress for non-TTY environments
                    if not use_tqdm:
                        logger.info(f"Processed {processed}/{total_docs} documents ({100*processed/total_docs:.1f}%)")

                    batch = []

            # Process remaining documents
            if batch:
                self._process_batch(batch)
                processed += len(batch)
                pbar.update(len(batch))

        logger.info(f"Successfully processed {processed} documents")

    def _build_with_training(self, total_docs: int, limit: Optional[int]):
        """Build vectorstore for IVF index (requires training)"""
        logger.info("Collecting all embeddings for IVF training...")

        all_embeddings = []
        all_texts = []
        all_metadatas = []
        processed = 0

        cursor = self.collection.find().limit(limit) if limit else self.collection.find()

        # Disable tqdm if not in a TTY (e.g., Docker)
        use_tqdm = sys.stderr.isatty()

        with tqdm(total=total_docs, desc="Collecting data", disable=not use_tqdm) as pbar:
            batch = []
            for doc in cursor:
                batch.append(doc)

                if len(batch) >= self.batch_size:
                    texts, embeddings, metadatas = self._prepare_batch(batch)
                    all_texts.extend(texts)
                    all_embeddings.append(embeddings)
                    all_metadatas.extend(metadatas)
                    processed += len(batch)
                    pbar.update(len(batch))

                    # Log progress for non-TTY environments
                    if not use_tqdm:
                        logger.info(f"Collected {processed}/{total_docs} documents ({100*processed/total_docs:.1f}%)")

                    batch = []

            # Process remaining
            if batch:
                texts, embeddings, metadatas = self._prepare_batch(batch)
                all_texts.extend(texts)
                all_embeddings.append(embeddings)
                all_metadatas.extend(metadatas)
                pbar.update(len(batch))

        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)

        # Train the index
        logger.info("Training IVF index...")
        self.faiss_index.train(all_embeddings)

        # Add all vectors
        logger.info("Adding vectors to index...")
        self.faiss_index.add(all_embeddings)

        # Store metadata
        self.faiss_metadata = [
            {'text': text, **metadata}
            for text, metadata in zip(all_texts, all_metadatas)
        ]

        logger.info(f"Successfully processed {len(all_texts)} documents")

    def _prepare_batch(self, batch: List[Dict[str, Any]]):
        """Prepare a batch and return texts, embeddings, and metadata"""
        texts = []
        metadatas = []
        skipped_count = 0

        for doc in batch:
            # Use abstract and title for embedding
            abstract = doc.get('abstract', '')
            title = doc.get('title', '')

            # Combine title and abstract
            text = f"{title}\n\n{abstract}" if title and abstract else (title or abstract)

            if not text:
                skipped_count += 1
                # Log first few skipped docs for debugging
                if skipped_count <= 3:
                    logger.warning(f"Skipping document - no abstract/title. Keys: {list(doc.keys())[:10]}")
                continue

            texts.append(text)

            # Prepare metadata (exclude large fields)
            metadata = {
                'paper_id': doc.get('paper_id', ''),
                'title': doc.get('title', ''),
                'authors': ', '.join(doc.get('authors', [])) if doc.get('authors') else '',
                'url': doc.get('url', ''),
                'topics': ', '.join(doc.get('topics', [])) if doc.get('topics') else '',
                'keywords': ', '.join(doc.get('keywords', [])) if doc.get('keywords') else '',
                'kwe_date': doc.get('kwe_date', ''),
            }
            metadatas.append(metadata)

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count}/{len(batch)} documents in batch (no abstract/title)")

        if not texts:
            return [], np.array([]), []

        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        return texts, embeddings, metadatas

    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of documents and add to FAISS"""
        texts, embeddings, metadatas = self._prepare_batch(batch)

        if len(texts) == 0:
            return

        # Add embeddings to index
        self.faiss_index.add(embeddings)

        # Store metadata and texts
        for text, metadata in zip(texts, metadatas):
            self.faiss_metadata.append({
                'text': text,
                **metadata
            })

    def _save_vectorstore(self):
        """Save the FAISS vectorstore to disk or MongoDB"""
        if self.storage_type == "mongodb":
            self._save_to_mongodb()
        else:
            self._save_to_file()

    def _save_to_file(self):
        """Save the FAISS vectorstore to local disk"""
        os.makedirs(self.vectorstore_path, exist_ok=True)
        index_path = os.path.join(self.vectorstore_path, "index.faiss")
        metadata_path = os.path.join(self.vectorstore_path, "metadata.pkl")

        faiss.write_index(self.faiss_index, index_path)
        logger.info(f"FAISS index saved to {index_path}")

        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.faiss_metadata, f)
        logger.info(f"FAISS metadata saved to {metadata_path}")

    def _save_to_mongodb(self):
        """Save the FAISS vectorstore to MongoDB using GridFS"""
        logger.info("Saving vectorstore to MongoDB...")

        # Serialize FAISS index to bytes using a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp_index:
            faiss.write_index(self.faiss_index, tmp_index.name)
            tmp_index.seek(0)
            with open(tmp_index.name, 'rb') as f:
                index_bytes = f.read()
            os.unlink(tmp_index.name)

        # Serialize metadata to bytes
        metadata_bytes = pickle.dumps(self.faiss_metadata)

        # Create vectorstore info document
        vectorstore_info = {
            'collection_name': self.collection_name,
            'index_type': self.index_type,
            'embedding_model': self.embedding_model_name,
            'total_vectors': self.faiss_index.ntotal,
            'embedding_dimension': self.faiss_index.d,
            'created_at': datetime.utcnow(),
            'num_documents': len(self.faiss_metadata)
        }

        # Delete old vectorstore if exists
        vectorstore_collection = self.db['vectorstores']
        old_doc = vectorstore_collection.find_one({'collection_name': self.collection_name})

        if old_doc:
            logger.info("Removing old vectorstore from MongoDB...")
            # Delete old GridFS files
            if 'index_file_id' in old_doc:
                self.fs.delete(old_doc['index_file_id'])
            if 'metadata_file_id' in old_doc:
                self.fs.delete(old_doc['metadata_file_id'])
            # Delete old document
            vectorstore_collection.delete_one({'_id': old_doc['_id']})

        # Save FAISS index to GridFS
        logger.info("Uploading FAISS index to GridFS...")
        index_file_id = self.fs.put(
            index_bytes,
            filename=f"{self.collection_name}_index.faiss",
            content_type="application/octet-stream",
            metadata={'type': 'faiss_index', 'collection': self.collection_name}
        )
        logger.info(f"FAISS index uploaded to GridFS with ID: {index_file_id}")

        # Save metadata to GridFS
        logger.info("Uploading metadata to GridFS...")
        metadata_file_id = self.fs.put(
            metadata_bytes,
            filename=f"{self.collection_name}_metadata.pkl",
            content_type="application/octet-stream",
            metadata={'type': 'metadata', 'collection': self.collection_name}
        )
        logger.info(f"Metadata uploaded to GridFS with ID: {metadata_file_id}")

        # Save vectorstore info with file references
        vectorstore_info['index_file_id'] = index_file_id
        vectorstore_info['metadata_file_id'] = metadata_file_id

        vectorstore_collection.insert_one(vectorstore_info)
        logger.info(f"Vectorstore info saved to 'vectorstores' collection")
        logger.info(f"Total size: {len(index_bytes) / 1024 / 1024:.2f} MB (index) + {len(metadata_bytes) / 1024 / 1024:.2f} MB (metadata)")

    def get_stats(self):
        """Get statistics about the vectorstore"""
        count = self.faiss_index.ntotal
        logger.info(f"FAISS index contains {count} vectors")
        return {"total_vectors": count}

    def cleanup(self):
        """Clean up resources"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


def main():
    parser = argparse.ArgumentParser(
        description="Build a FAISS vectorstore from MongoDB documents with keywords"
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="MongoDB collection name containing documents with keywords"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["flat", "ivf", "hnsw"],
        default="flat",
        help="Type of FAISS index (flat=exact, ivf=fast approximate, hnsw=very fast approximate)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformers model to use for embeddings"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./vectorstore_data",
        help="Path to save the vectorstore (default: ./vectorstore_data)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing documents (default: 100)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (optional)"
    )
    parser.add_argument(
        "--storage-type",
        type=str,
        choices=["file", "mongodb"],
        default="file",
        help="Where to store the vectorstore (file or mongodb)"
    )

    args = parser.parse_args()

    try:
        builder = FAISSVectorStoreBuilder(
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            vectorstore_path=args.output_path,
            batch_size=args.batch_size,
            index_type=args.index_type,
            storage_type=args.storage_type
        )

        builder.build_vectorstore(limit=args.limit)
        stats = builder.get_stats()

        logger.info("=" * 50)
        logger.info("FAISS vectorstore build complete!")
        logger.info(f"Total vectors: {stats['total_vectors']}")
        logger.info(f"Index type: {args.index_type}")
        logger.info(f"Storage type: {args.storage_type}")
        if args.storage_type == "file":
            logger.info(f"Location: {args.output_path}")
        else:
            logger.info(f"Location: MongoDB ({DB_NAME}.vectorstores)")
        logger.info("=" * 50)

        builder.cleanup()

    except Exception as e:
        logger.error(f"Error building vectorstore: {e}")
        raise


if __name__ == "__main__":
    main()
