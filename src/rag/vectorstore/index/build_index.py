from documents import get_all_documents
from atlas import connect_to_atlas

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from tqdm import tqdm
from argparse import ArgumentParser

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

    parser = ArgumentParser(description="Build index for vectorstore.")
    parser.add_argument(
        "--path",
        type=str,
        default="./saved_indices",
        help="Path to save the index files."
    )
    args = parser.parse_args()
    path = args.path


    for i in tqdm(range(0, len(documents), vectorstore_size), desc="Constructing shard indices..."):
        documents_size = min(vectorstore_size, len(documents) - i)
        index.construct_shard_index(
            embedding=embedding_module,
            documents_list=documents[i:i + documents_size],
            nlist=100,
            nprobe=10,
            directory=f"{path}/index_{i}"
        )
