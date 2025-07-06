from uuid import uuid4

from concurrent.futures import ThreadPoolExecutor, as_completed

from argparse import ArgumentParser
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

from documents import get_all_documents
from vectorstore import build_vectorstore_shard

if __name__ == '__main__':
    documents = get_all_documents()
    uuids = [str(uuid4()) for _ in range(10)]

    embedding_module = OpenAIEmbeddings(model="text-embedding-3-small")
    parser = ArgumentParser(description="Build a vector store shard from a list of documents.")
    parser.add_argument("--save_path", 
                        type=str, 
                        default="./saved_vectorstore_shards", 
                        help="The path to save the vectorstore shards."

                        )
    parser.add_argument("--name",
                        type=str,
                        default="shard",
                        help="The base name of the vectorstore shard."
                        )
    
    args = parser.parse_args()
    save_path = args.save_path
    base_name = args.name

    batch_size = 64
    vectorstore_size = batch_size * 500

    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i in range(0, len(documents), vectorstore_size):
            # Determine the size of the current batch
            documents_size = min(vectorstore_size, len(documents) - i)
            batch_documents = documents[i:i + documents_size]

            name = f"{base_name}_{i}"
            # Submit the task to construct the shard index
            futures.append(
                executor.submit(
                    build_vectorstore_shard,
                    embedding=embedding_module,
                    documents_list=batch_documents,
                    batch_size=batch_size,
                    start=i,
                    name=name,
                    path=save_path
                )
            )
        for future in as_completed(futures):
            try:
                shard = future.result()
                print(f"Shard created with size: {shard.get_size()}")
            except Exception as e:
                print(f"Error creating shard: {e}")



