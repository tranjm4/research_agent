from pymongo.mongo_client import MongoClient

from langchain_ollama import ChatOllama
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()
import os

from tqdm import tqdm

def connect_to_atlas():
    """
    Connect to the MongoDB Atlas database.
    
    Returns:
        MongoClient: The MongoDB client connected to the Atlas database.
    """
    # Replace with your actual connection string
    mongo_uri = os.getenv("ATLAS_URI")
    client = MongoClient(mongo_uri)
    try:
        client.admin.command('ping')  # Test the connection
        print("Connected to MongoDB Atlas successfully.")
    except Exception as e:
        print(f"Failed to connect to MongoDB Atlas: {e}")
        raise e
    return client

def get_tagging_prompt():
    """
    Get the prompt for tagging documents.
    
    Returns:
        str: The prompt for tagging documents.
    """
    return (
        """
        You are an expert in tagging research papers. 
        Your task is to generate a list of single-word tags for the given document content. 
        Include up to 20 tags, separated by commas.
        Convert any plural nouns to singular form if its meaning is the same.
        
        Do not include any additional text or explanations. Only respond with the Python list of tags.
        
        Return the tags in a Python list format like this:
        [tag1, tag2, tag3, ...]
        
        EXAMPLES:
        input: 'Analyzing network structural connectivity is crucial for understanding dynamics and functions 
        of complex networks across disciplines. In many networks, structural connectivity is not observable, 
        which requires to be inferred via causal inference methods. Among them, transfer entropy (TE) is one 
        of the most broadly applied causality measure due to its model-free property. However, TE often faces 
        the curse of dimensionality in high-dimensional probability estimation, and the relation between the 
        inferred causal connectivity and the underlying structural connectivity remains poorly understood. 
        Here we address these issues by proposing a pairwise time-delayed transfer entropy (PTD-TE) method. 
        We theoretically establish a quadratic relationship between PTD-TE values and node coupling strengths, 
        and demonstrate its immunity to dimensionality issues and broad applicability. Tests on biological 
        neuronal networks, nonlinear physical systems, and electrophysiological data show PTD-TE achieves 
        consistent, high-performance reconstructions. Compared to a bunch of existing approaches for network 
        connectivity reconstruction, PTD-TE outperforms these methods across various network systems in accuracy 
        and robustness against noise. Our framework provides a scalable, model-agnostic tool for structural 
        connectivity inference in nonlinear real-world networks.'
        
        output: [network, connectivity, inference, causality, transfer entropy, time series,
        nonlinear, reconstruction, dimensionality, information theory, scalability, noise, robustness,
        coupling, dynamics, model-free, neuronal, biological, electrophysiology, system]
        """
        
    )
    
def load_llm():
    """
    Load the LLM for tagging documents.
    """
    return ChatOllama(model="llama3.2", temperature=0.1, max_tokens=1000,)

def load_chain():
    """
    Load the chain for tagging documents.
    
    Returns:
        ChatOllama: The LLM instance for tagging documents.
    """
    tagging_prompt = get_tagging_prompt()
    
    llm = load_llm()
    
    system_template = SystemMessagePromptTemplate.from_template(tagging_prompt)
    human_template = HumanMessagePromptTemplate.from_template("{input}")
    
    chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])
    
    chain = RunnableSequence(
        lambda x: {"input": x["input"]},
        chat_prompt | llm 
        | RunnableLambda(lambda x: parse_tag_output(x.content))
    )
    
    return chain

def parse_tag_output(content):
    """
    Parses the output from the LLM to extract tags.
    
    Args:
        content (str): The output string from the LLM containing tags.
        
    Returns:
        list: A list of tags extracted from the output.
    """
    # Assuming the output is in the format: [tag1, tag2, tag3, ...]
    content = content.strip()
    if content.startswith("[") and content.endswith("]"):
        tags = content[1:-1].split(",")
        return [tag.strip() for tag in tags]
    return []

def tag_document(doc, chain, db):
    """
    Tag a single document with a specific tag using the LLM.
    Update the document in the MongoDB collection with the generated tags.
    
    Args:
        doc (dict): The document to tag.
        llm (ChatOllama): The LLM instance for tagging.
        collection: The MongoDB collection where the document is stored.
    
    Returns:
        dict: The updated document with the tag.
    """
    # Generate a tag using the LLM
    tags = chain.invoke({"input": doc["abstract"]})
    
    # Find duplicate document in other collections
    cross_topics = doc.get("topic", [])
    for topic in cross_topics:
        collection = db[topic]
        
        collection.update_one(
            {"paper_id": doc["paper_id"]},
            {"$set": {"tags": tags, "tagged": True}},
        )
    return

def tag_documents():
    """
    Main function to tag documents in the MongoDB collection.
    """
    client = connect_to_atlas()
    db = client["arxiv_db"]
    
    # Load the LLM for tagging
    chain = load_chain()
    
    # Process documents in the collection
    for collection_name in tqdm(db.list_collection_names(), desc="Processing collections"):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            collection = db[collection_name]
            
            # Submit the tagging task for each document in the collection
            for doc in tqdm(collection.find(), desc=f"Tagging documents in {collection_name}"):
                if not doc.get("tagged", False):
                    futures.append(executor.submit(tag_document, doc, chain, db))
            
            for future in tqdm(as_completed(futures), desc="Waiting for tagging results"):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error tagging document: {e}")
            
            executor.shutdown(wait=True)
                    
    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     futures = []
        
    #     for collection_name in db.list_collection_names():
    #         collection = db[collection_name]
            
    #         # Submit the tagging task for each document in the collection
    #         for doc in collection.find():
    #             futures.append(executor.submit(tag_document, doc, llm, collection))
                
    #     for future in as_completed(futures):
    #         result = future.result()
    
        # for doc in db["papers"].find():
        #     futures.append(executor.submit(tag_document, doc, llm, db["papers"]))
        
        # for future in as_completed(futures):
        #     result = future.result()
        #     if result:
        #         print(f"Tagged document: {result}")

if __name__ == "__main__":
    tag_documents()