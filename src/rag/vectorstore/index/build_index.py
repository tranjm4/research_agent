from documents import get_all_documents
from atlas import connect_to_atlas

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

import index

def _get_documents():
    """
    Retrieve all documents from the MongoDB Atlas database.
    
    Returns:
        list: A list of LangChain Document objects.
    """
    client = connect_to_atlas()
    database_name = "arxiv_db"
    documents = get_all_documents(client, database_name)
    
    return documents


if __name__ == '__main__':
    """
    Script to build the index.
    """
    documents = _get_documents()
    embedding_module = OpenAIEmbeddings(model="text-embedding-3-small")

    batch_size = 64
    vectorstore_size = batch_size * 500

    for i in tqdm(range(0, len(documents), vectorstore_size), desc="Constructing shard indices..."):
        documents_size = min(vectorstore_size, len(documents) - i)
        index.construct_shard_index(
            name=f"shard_index_{i}",
            embedding=embedding_module,
            documents_list=documents[i:i + documents_size],
            nlist=100,
            nprobe=10,
            directory="./saved_indices"
        )
