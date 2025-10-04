"""
File: src/backend/data/vectorstore/load_vectorstore.py

This module provides functionality to load and search the FAISS vectorstore from MongoDB.
"""

import faiss
import pickle
import tempfile
import os
from typing import List, Dict, Any, Optional
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.errors import PyMongoError
from gridfs import GridFS
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class MongoVectorStore:
    """Load and search FAISS vectorstore stored in MongoDB"""

    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the MongoDB VectorStore loader

        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
            collection_name: Name of the collection the vectorstore was built from
            embedding_model: Optional override for embedding model name
        """
        self.collection_name = collection_name

        # Connect to MongoDB
        self.client = MongoClient(mongo_uri)
        try:
            self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            raise PyMongoError(f"Failed to connect to MongoDB: {e}")

        self.db = self.client[db_name]
        self.fs = GridFS(self.db)

        # Load vectorstore info
        vectorstore_collection = self.db['vectorstores']
        self.vectorstore_info = vectorstore_collection.find_one(
            {'collection_name': collection_name}
        )

        if not self.vectorstore_info:
            raise ValueError(f"No vectorstore found for collection '{collection_name}'")

        logger.info(f"Found vectorstore: {self.vectorstore_info['total_vectors']} vectors, "
                   f"created at {self.vectorstore_info['created_at']}")

        # Load FAISS index from GridFS
        self._load_index()

        # Load metadata from GridFS
        self._load_metadata()

        # Load embedding model
        model_name = embedding_model or self.vectorstore_info['embedding_model']
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)

    def _load_index(self):
        """Load FAISS index from GridFS"""
        index_file_id = self.vectorstore_info['index_file_id']
        logger.info(f"Loading FAISS index from GridFS (ID: {index_file_id})...")

        # Read index bytes from GridFS
        index_data = self.fs.get(index_file_id).read()

        # Write to temporary file and load with FAISS
        with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp:
            tmp.write(index_data)
            tmp_path = tmp.name

        try:
            self.index = faiss.read_index(tmp_path)
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
        finally:
            os.unlink(tmp_path)

    def _load_metadata(self):
        """Load metadata from GridFS"""
        metadata_file_id = self.vectorstore_info['metadata_file_id']
        logger.info(f"Loading metadata from GridFS (ID: {metadata_file_id})...")

        # Read metadata bytes from GridFS
        metadata_bytes = self.fs.get(metadata_file_id).read()

        # Deserialize
        self.metadata = pickle.loads(metadata_bytes)
        logger.info(f"Metadata loaded: {len(self.metadata)} documents")

    def keyword_search(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search on MongoDB collection with text index
        Searches across abstract, title, and keywords fields

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of search results with text scores
        """
        collection = self.db[self.collection_name]

        # MongoDB text search across abstract, title, and keywords
        results = []
        cursor = collection.find(
            {'$text': {'$search': query}},
            {'score': {'$meta': 'textScore'}, 'abstract': 1, 'title': 1, 'authors': 1,
             'url': 1, 'topics': 1, 'paper_id': 1, 'keywords': 1}
        ).sort([('score', {'$meta': 'textScore'})]).limit(top_k)

        for doc in cursor:
            # Use abstract + title as the text (same as vectorstore)
            abstract = doc.get('abstract', '')
            title = doc.get('title', '')
            text = f"{title}\n\n{abstract}" if title and abstract else (title or abstract)

            results.append({
                'score': doc.get('score', 0),
                'text': text,
                'metadata': {
                    'title': doc.get('title', ''),
                    'authors': doc.get('authors', []),
                    'url': doc.get('url', ''),
                    'topics': doc.get('topics', []),
                    'paper_id': doc.get('paper_id', ''),
                    'keywords': doc.get('keywords', [])
                }
            })

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filter_by: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and keyword matching with reranking

        Args:
            query: Search query text
            top_k: Number of final results to return
            vector_weight: Weight for vector search scores (0-1)
            keyword_weight: Weight for keyword search scores (0-1)
            filter_by: Optional metadata filters

        Returns:
            Reranked list of search results
        """
        # Get results from both searches (fetch more for reranking)
        fetch_k = top_k * 3

        vector_results = self.search(query, top_k=fetch_k, filter_by=filter_by)
        keyword_results = self.keyword_search(query, top_k=fetch_k)

        # Normalize scores to 0-1 range
        def normalize_scores(results):
            if not results:
                return results
            max_score = max(r['score'] for r in results)
            min_score = min(r['score'] for r in results)
            score_range = max_score - min_score if max_score != min_score else 1

            for r in results:
                r['normalized_score'] = (r['score'] - min_score) / score_range
            return results

        vector_results = normalize_scores(vector_results)
        keyword_results = normalize_scores(keyword_results)

        # Combine results using RRF (Reciprocal Rank Fusion) + weighted scores
        combined = {}

        # Add vector results
        for rank, result in enumerate(vector_results, 1):
            key = result['metadata'].get('paper_id', '') + '_' + result['text'][:50]
            rrf_score = 1 / (60 + rank)  # RRF with k=60
            combined[key] = {
                **result,
                'combined_score': vector_weight * result.get('normalized_score', 0) + rrf_score,
                'vector_rank': rank
            }

        # Add keyword results
        for rank, result in enumerate(keyword_results, 1):
            key = result['metadata'].get('paper_id', '') + '_' + result['text'][:50]
            rrf_score = 1 / (60 + rank)

            if key in combined:
                # Merge scores if already exists
                combined[key]['combined_score'] += keyword_weight * result.get('normalized_score', 0) + rrf_score
                combined[key]['keyword_rank'] = rank
            else:
                combined[key] = {
                    **result,
                    'combined_score': keyword_weight * result.get('normalized_score', 0) + rrf_score,
                    'keyword_rank': rank
                }

        # Sort by combined score and return top_k
        reranked = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
        return reranked[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_by: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_by: Optional metadata filters (e.g., {'paper_id': 'xyz', 'topics': 'machine learning'})

        Returns:
            List of search results with scores and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search FAISS index (get more results for filtering)
        search_k = top_k * 3 if filter_by else top_k
        scores, indices = self.index.search(query_embedding, search_k)

        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue

            metadata = self.metadata[idx].copy()

            # Apply filters if specified
            if filter_by:
                match = True
                for key, value in filter_by.items():
                    meta_value = metadata.get(key, '')
                    # Case-insensitive substring match
                    if value.lower() not in str(meta_value).lower():
                        match = False
                        break
                if not match:
                    continue

            result = {
                'score': float(score),
                'text': metadata.pop('text'),
                'metadata': metadata
            }
            results.append(result)

            # Stop once we have enough results
            if len(results) >= top_k:
                break

        return results

    def search_by_paper(
        self,
        paper_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks from a specific paper

        Args:
            paper_id: The paper ID to search for
            limit: Maximum number of chunks to return

        Returns:
            List of chunks from the paper
        """
        results = []
        for metadata in self.metadata:
            if metadata.get('paper_id') == paper_id:
                result = {
                    'text': metadata.get('text'),
                    'metadata': {k: v for k, v in metadata.items() if k != 'text'}
                }
                results.append(result)

                if len(results) >= limit:
                    break

        return results

    def get_papers_by_topic(
        self,
        topic: str,
        limit: int = 20
    ) -> List[Dict[str, str]]:
        """
        Get unique papers that match a specific topic

        Args:
            topic: Topic to filter by (case-insensitive substring match)
            limit: Maximum number of papers to return

        Returns:
            List of unique papers with metadata
        """
        papers = {}

        for metadata in self.metadata:
            topics = metadata.get('topics', '')
            if topic.lower() in topics.lower():
                paper_id = metadata.get('paper_id')
                if paper_id and paper_id not in papers:
                    papers[paper_id] = {
                        'paper_id': paper_id,
                        'title': metadata.get('title', ''),
                        'authors': metadata.get('authors', ''),
                        'url': metadata.get('url', ''),
                        'topics': metadata.get('topics', ''),
                        'keywords': metadata.get('keywords', '')
                    }

                if len(papers) >= limit:
                    break

        return list(papers.values())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vectorstore

        Returns:
            Dictionary with vectorstore statistics
        """
        # Count unique papers and topics
        unique_papers = set()
        unique_topics = set()

        for metadata in self.metadata:
            if metadata.get('paper_id'):
                unique_papers.add(metadata.get('paper_id'))
            if metadata.get('topics'):
                for topic in metadata.get('topics', '').split(', '):
                    if topic:
                        unique_topics.add(topic)

        return {
            'total_documents': self.index.ntotal,
            'total_papers': len(unique_papers),
            'total_topics': len(unique_topics),
            'embedding_dimension': self.index.d,
            'index_type': self.vectorstore_info.get('index_type'),
            'created_at': self.vectorstore_info.get('created_at')
        }

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Load from environment
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chunked_docs")

    # Create vectorstore
    vectorstore = MongoVectorStore(
        mongo_uri=MONGO_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME
    )

    # Get stats
    stats = vectorstore.get_stats()
    print(f"\nVectorstore Stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total papers: {stats['total_papers']}")
    print(f"  Total topics: {stats['total_topics']}")
    print(f"  Index type: {stats['index_type']}")

    # Example search
    query = "machine learning transformers"
    print(f"\nSearching for: '{query}'")
    results = vectorstore.search(query, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Title: {result['metadata']['title']}")
        print(f"   Text: {result['text'][:200]}...")

    vectorstore.close()
