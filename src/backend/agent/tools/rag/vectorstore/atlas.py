"""
File: src/rag/vectorstore/atlas.py

This module provides basic helpers to connect to MongoDB Atlas
"""

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from dotenv import load_dotenv
import os
load_dotenv()

def connect_to_atlas():
    """
    Connect to MongoDB Atlas using the connection string from the environment variable.
    """
    atlas_uri = os.getenv("ATLAS_URI")
    client = MongoClient(atlas_uri, server_api=ServerApi('1'))

    try:
        # Attempt to get the server information to verify the connection
        client.admin.command('ping')
        print("\n[Atlas] Successfully connected to the database.\n")
    except Exception as e:
        print(f"[Atlas] Failed to connect to the database: {e}")
        raise ConnectionError("Could not connect to MongoDB Atlas. Please check connection string.")
    return client