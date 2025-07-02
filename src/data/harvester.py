"""
HARVESTER

Interacts with arXiv's Open Archives Initiative (OAI) to harvest archive metadata.
"""

import requests
import xml.etree.ElementTree as ET

from tqdm import tqdm

from datetime import datetime, timezone, timedelta
from time import sleep

from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os

from concurrent.futures import ThreadPoolExecutor

REQUEST_BASE_URL = "https://oaipmh.arxiv.org/oai?"

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}

import argparse
DEBUG = False

def connect_to_atlas():
    load_dotenv()
    atlas_uri = os.getenv("ATLAS_URI")
    client = MongoClient(atlas_uri)
    try:
        client.admin.command("ping")
        print("[ATLAS]\t\tSuccessfully connected to database")
        return client
    except:
        print("[ATLAS] [ERROR]\tFailed to connect to database")
        return None

def send_request(client, resumption_token = None, from_date = None, until_date = None):
    params = build_params(resumption_token, from_date, until_date)
            
    dprint(params)
    
    # make request
    response = requests.get(
        REQUEST_BASE_URL,
        params=params
    )
    
    root = ET.fromstring(response.text)
    
    # process response
    records = root.findall(".//oai:record", namespaces=NS)
    
    dprint(len(records))
    
    extract_metadata(records, client, from_date, until_date)
    
    resumption_token = root.find(".//oai:resumptionToken", namespaces=NS)
    if resumption_token is not None:
        resumption_token = resumption_token.text
        
    dprint(resumption_token)
    
    return resumption_token

def build_params(resumption_token, from_date, until_date):
    params = {}
    if resumption_token:
        # do not include from and until parameters if resumption token
        params["resumptionToken"] = resumption_token
    else:
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv"
        }
        if from_date:
            params["from"] = from_date
        if until_date:
            params["until"] = until_date
            
    return params
    
def load_last_timestamp(client):
    last_timestamp = client["arxiv_db"]["metadata"].find_one({"key": "last_harvest"})
    if last_timestamp:
        return last_timestamp["value"]
    else:
        return None

def update_last_timestamp(client):
    current_datetime_str = get_current_datetime()
    client["arxiv_db"]["metadata"].update_one(
        {"key": "last_harvest"},
        {"$set": {"value": current_datetime_str}},
        upsert=True
    )
    
    return

def get_current_datetime():
    current_utc = datetime.now(timezone.utc)
    current_utc_str = current_utc.strftime("%Y-%m-%d")
    
    return current_utc_str

def extract_metadata(records, client, from_date, until_date):
    for record in tqdm(records, desc=f"Extracting metadata [{from_date} ==> {until_date}]", leave=True):
        header = record.find("oai:header", NS)
        identifier = header.find("oai:identifier", NS).text
        datestamp = header.find("oai:datestamp", NS).text
        topics = header.findall("oai:setSpec", NS)
        main_topics = list(set(topic.text.split(":")[0] for topic in topics))
        sub_topics = []
        for topic in topics:
            sub_topics.append(".".join(topic.text.split(":")[1:]))
        sub_topics = list(set(sub_topics))
        
        metadata = record.find("oai:metadata", NS)
        arxiv_meta = metadata.find("arxiv:arXiv", NS)
        
        paper_id = arxiv_meta.find("arxiv:id", NS).text
        abstract = arxiv_meta.find("arxiv:abstract", NS).text.strip()
        created = arxiv_meta.find("arxiv:created", NS).text
        updated = arxiv_meta.find("arxiv:updated", NS)
        if updated:
            updated = updated.text
        else:
            updated = created
        
        authors = []
        for author in arxiv_meta.findall("arxiv:authors/arxiv:author", NS):
            keyname = author.find("arxiv:keyname", NS).text
            forenames = author.find("arxiv:forenames", NS)
            if forenames:
                forenames = forenames.text
            else:
                forenames = ""
            full_name = f"{forenames},{keyname}".strip()
            authors.append(full_name)
            
        # insert into Atlas database
        document = {
            "url": f"https://arxiv.org/abs/{paper_id}",
            "paper_id": paper_id,
            "identifier": identifier,
            "datestamp": datestamp,
            "created_date": created,
            "updated_date": updated,
            "abstract": abstract,
            "topic": main_topics,
            "subtopics": sub_topics,
        }
        insert_to_database(document, client)
        
    return
        
def insert_to_database(document, client):
    topics = document["topic"]
    database = client["arxiv_db"]
    for topic in topics:
        database[topic].insert_one(document)
    
    return

def generate_week_ranges(start_date, end_date):
    current = start_date
    while current < end_date:
        week_end = min(current + timedelta(days=7), end_date)
        yield current.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")
        current = week_end

def harvest(from_date, until_date):
    mongo_client = connect_to_atlas()
    if mongo_client == None:
        return
    
    num_paginates = 1
    resumption_token = None
    while True:
        resumption_token = send_request(mongo_client, 
                                        resumption_token=resumption_token, 
                                        from_date=from_date, 
                                        until_date=until_date)
        if resumption_token is None:
            break
        num_paginates += 1
        sleep(3)

def harvest_threading():
    client = connect_to_atlas()
    end = datetime.now() # set end to today
    start = load_last_timestamp(client)
    client.close()
    if start == None:
        start = end - timedelta(days=365 * 2) # by default, set start point to 4 years ago
        
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for from_str, until_str in generate_week_ranges(start, end):
            futures.append(executor.submit(harvest, from_str, until_str))
            
        for future in futures:
            future.result()
            
def dprint(s):
    if DEBUG == True:
        print(s)
            
def parse_args():
    global DEBUG
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, help="Custom greeting message", action="store_true")
    args = parser.parse_args()
    DEBUG = args.debug
    print("-"*20)
    print(f"DEBUG MODE SET TO {DEBUG}")
    print("-"*20)
    
if __name__ == "__main__":
    parse_args()
    harvest_threading()
    
# TODO: Ran out of storage at weeks 2025-01-13 => 01-20 => 01-27 => 02-03