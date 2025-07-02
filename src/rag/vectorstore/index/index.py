"""
File: src/rag/vectorstore/index.py

This module provides functionality to create the vector store index using FAISS and OpenAI embeddings.
"""

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from atlas import connect_to_atlas
from dotenv import load_dotenv
import os
load_dotenv()

import documents

import numpy as np

from uuid import uuid4
from tenacity import retry, stop_after_attempt, wait_random_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

def construct_shard_index(name, embedding, documents_list, nlist=100, nprobe=10, directory="./saved_indices"):
    """
    Construct and save a shard index for the given name using the specified embedding model.

    Args:
        name (str): The name of the shard index to be saved.
        embedding (OpenAIEmbeddings): The embedding model to use for vectorization
        nlist (int): The number of clusters for the FAISS index (default is 100).
        nprobe (int): The number of clusters to search during query time (default is
        directory (str): The directory where the index will be saved (default is "./saved_indices").

    Returns:
        (FAISS): The constructed FAISS index.
    """
    index = _init_empty_index(embedding, nlist=nlist, nprobe=nprobe)
    training_vectors = get_training_vectors(documents_list, embedding)

    index.train(training_vectors)

    save_shard_index(index, name, directory)
    return index


def _init_empty_index(embedding, nlist, nprobe):
    """
    Initialize an empty FAISS index with the given embedding dimensions.

    Args:
        embedding (OpenAIEmbeddings): The embedding model to use for vectorization.
        nlist (int): The number of clusters for the FAISS index (default is 100).

    Returns:
        (faiss.IndexFlatIVF): An empty FAISS index ready for training and adding vectors.
    """
    dim = embedding.embed_query("test").shape[0]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexFlatIVF(quantizer, dim, nlist, faiss.METRIC_L2)

    index.nprobe = nprobe  # Set the number of clusters to search during query time

    return index

def get_training_vectors(documents_list, embedding):
    """
    Generate training vectors for the FAISS index from a list of documents.

    Args:
        documents_list (list): A list of LangChain Document objects.
        embedding (OpenAIEmbeddings): The embedding model to use for vectorization.

    Returns:
        (np.ndarray): An array of vectors representing the documents.
    """
    training_size = len(documents_list) // 3
    training_documents = documents_list[:training_size]

    training_vectors = embedding.embed_documents([doc.page_content for doc in training_documents])
    training_vectors = np.array(training_vectors, dtype=np.float32)

    return training_vectors
    

def save_shard_index(shard_index, name, path) -> None:
    """
    Save the FAISS shard index to a file.

    Args:
        shard_index (FAISS): The FAISS index to be saved.
        name (str): The name of the shard index.
        path (str): The directory where the index will be saved.
    """

    if os.path.exists(path) == False:
        os.makedirs(path)
    faiss.write_index(shard_index, f"{path}/{name}.faiss")

    return
