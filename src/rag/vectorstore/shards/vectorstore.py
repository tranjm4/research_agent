import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

from tenacity import retry, stop_after_attempt, wait_random_exponential

from uuid import uuid4

import os

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class VectorStoreShard:
    def __init__(self, embedding_module, index, capacity):
        self.vector_store = FAISS(
            embedding_function=embedding_module,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        self.size = 0
        self.capacity = capacity

    def get_size(self):
        return self.size

    def save_local(self, *args, **kwargs):
        print("Saving vector store...")
        self.vector_store.save_local(*args, **kwargs)

    def add_documents(self, documents_batch, uuids):
        self.vector_store = add_documents_with_retry(self.vector_store, documents_batch, uuids)
        self.size += len(documents_batch)

    def is_full(self):
        return self.size >= self.capacity
  
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def add_documents_with_retry(vector_store, batch, batch_uuids):
    try:
        vector_store.add_documents(documents=batch, uuids=batch_uuids)

        return vector_store
    except Exception as e:
        print("Failed batch")
        print(e)

def load_index(start):
    """
    Load the FAISS index from the specified directory.

    Args:
        start (int): The starting index for the shard.

    Returns:
        faiss.IndexFlatIVF: The loaded FAISS index.
    """
    index_path = f"./saved_indices/index_{start}"
    print(f"Loading index from {index_path}...")
    
    # Load the FAISS index
    index_path = f"{index_path}/index_{start}.faiss"
    assert os.path.exists(index_path), f"Index file {index_path} does not exist."
    index = faiss.read_index(index_path)

    return index.index

def log_progress(name, i, total):
    """
    Log the progress of the vector store shard creation.

    Args:
        name (str): The name of the shard.
        i (int): The current index in the batch.
        total (int): The total number of documents.
    """
    progress = i // 50
    if progress % 50 == 0:
        # Calculate the percentage of documents processed
        percentage = (i / total) * 100
        # Print the progress
        print(f"[{name}]\tProgress: {percentage}% of {total} documents processed.")

def build_vectorstore_shard(embedding, documents_list, batch_size, start, name, path="./saved_vectorstore_shards"):
    """
    Build a vector store shard from a list of documents.

    Args:
        documents_list (list): List of LangChain Document objects.
        batch_size (int): The size of each batch to be added to the vector store.

    Returns:
        VectorStoreShard: A shard containing the vector store and its metadata.
    """
    index = load_index(start)

    shard = VectorStoreShard(embedding, index, capacity=batch_size)

    for i in range(0, len(documents_list), batch_size):
        batch = documents_list[i:i + batch_size]
        uuids = [str(uuid4()) for _ in range(len(batch))]
        shard.add_documents(batch, uuids)

        log_progress(name, i // batch_size, len(documents_list))

    save_dir = f"{path}/{name}"
    shard.save_local(save_dir, name)


    return shard

def load_shards(embedding_module, path="./saved_shards", k=3):
    """
    Load all vector store shards from the specified directory.

    Args:
        path (str): The directory where the vector store shards are saved.

    Returns:
        list: A list of VectorStoreShard objects.
    """
    shards = []
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, path)
    for dir_name in os.listdir(path):
        dir_path = os.path.join(path, dir_name)
        dir_path = os.path.abspath(dir_path)
        if os.path.isdir(dir_path):
            shard = FAISS.load_local(dir_path,
                                     embeddings=embedding_module,
                                     allow_dangerous_deserialization=True)
            shard = shard.as_retriever(search_type="similarity", search_kwargs={"k": k})
            shards.append(shard)
        else:
            raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")

    return shards

if __name__ == "__main__":
    embedding_module = OpenAIEmbeddings(model="text-embedding-3-small")
    shards = load_shards(embedding_module)
    from pprint import pprint
    pprint(shards)