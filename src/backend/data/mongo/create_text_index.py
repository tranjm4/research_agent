"""
File: src/backend/data/mongo/create_text_index.py

This script creates a MongoDB text index on the KWE collection to enable keyword-based search.
The index is created on abstract, title, and keywords fields for hybrid search functionality.
"""

from pymongo import MongoClient, TEXT
from pymongo.errors import PyMongoError
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
KWE_COLLECTION = os.getenv("KWE_COLLECTION")

if not all([MONGO_URI, DB_NAME, KWE_COLLECTION]):
    raise ValueError("Missing required environment variables: MONGO_URI, DB_NAME, or KWE_COLLECTION")


def create_text_index():
    """Create a text index on the KWE collection for keyword search"""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        client.admin.command("ping")
        logger.info("Successfully connected to MongoDB")

        db = client[DB_NAME]
        collection = db[KWE_COLLECTION]

        # Check if text index already exists
        existing_indexes = collection.index_information()
        has_text_index = False
        text_index_name = None

        for index_name, index_info in existing_indexes.items():
            key = index_info.get('key', [])
            # key is a list of tuples like [('field', 'text'), ...]
            if any(field_type == 'text' for _, field_type in key):
                has_text_index = True
                text_index_name = index_name
                break

        if has_text_index:
            logger.info(f"Text index already exists on collection '{KWE_COLLECTION}': {text_index_name}")
            logger.info("Dropping existing text index to recreate with updated weights...")
            collection.drop_index(text_index_name)
            logger.info(f"Dropped existing text index: {text_index_name}")

        # Create text index with weighted fields
        # Higher weights give more importance to that field in search ranking
        logger.info(f"Creating text index on collection '{KWE_COLLECTION}'...")

        index_result = collection.create_index(
            [
                ('title', TEXT),
                ('abstract', TEXT),
                ('keywords', TEXT)
            ],
            name='text_search_index',
            weights={
                'title': 10,      # Title matches are most important
                'keywords': 5,    # Keyword matches are moderately important
                'abstract': 2     # Abstract matches are less important (but still relevant)
            },
            default_language='english'
        )

        logger.info(f"Text index created successfully: {index_result}")

        # Verify index creation
        indexes = collection.index_information()
        logger.info(f"Current indexes on '{KWE_COLLECTION}':")
        for index_name, index_info in indexes.items():
            logger.info(f"  - {index_name}: {index_info}")

        # Get collection stats
        doc_count = collection.count_documents({})
        logger.info(f"Collection contains {doc_count} documents")

        client.close()
        logger.info("MongoDB connection closed")

        return True

    except PyMongoError as e:
        logger.error(f"MongoDB error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error creating text index: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting text index creation...")
    success = create_text_index()

    if success:
        logger.info("✓ Text index setup completed successfully")
        logger.info("The collection is now ready for hybrid search (vector + keyword)")
    else:
        logger.error("✗ Failed to create text index")
        exit(1)
