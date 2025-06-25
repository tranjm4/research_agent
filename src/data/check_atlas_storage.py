from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os

def connect_to_atlas():
    load_dotenv()
    client = MongoClient(os.getenv("ATLAS_URI"))
    
    try:
        client.admin.command("ping")
        print("[ATLAS]: Client connection successful")
    except:
        print("[ATLAS] [ERROR] Client connection failed")
        raise Exception
    
    return client

def print_stats():
    client = connect_to_atlas()
    
    db = client["arxiv_db"]
    
    collections = db.list_collection_names()
    
    total_document_count = 0
    total_logical_size = 0
    total_index_size = 0
    total_storage_size = 0

    for coll_name in collections:
        stats = db.command("collStats", coll_name)
        print(f"Collection: {coll_name}")
        print(f"  Document count: {stats['count']}")
        print(f"  Logical size (size): {stats['size'] / (1024 * 1024):.2f} MB")
        print(f"  Index size: {stats['totalIndexSize'] / (1024 * 1024):.2f} MB")
        print(f"  Storage size (on disk): {stats['storageSize'] / (1024 * 1024):.2f} MB")
        print()
        
        total_document_count += stats["count"]
        total_logical_size += stats["size"]
        total_index_size += stats["totalIndexSize"]
        total_storage_size += stats["storageSize"]
    
    print("-"*50)
    print("DATABASE STATS")
    print("-"*50)
    
    print(f"  Document count: {total_document_count}")
    print(f"  Logical size (size): {total_logical_size / (1024 * 1024):.24} MB")
    print(f"  Index size: {total_index_size / (1024 * 1024):.2f} MB")
    print(f"  Storage size (on disk) {total_storage_size / (1024 * 1024):.2f} MB")
        
    client.close()
    
    return

if __name__ == '__main__':
    print_stats()