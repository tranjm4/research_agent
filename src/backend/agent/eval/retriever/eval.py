"""
File: src/backend/agent/eval/search/eval.py

This file contains the evaluation script for the search model.

Evaluation metrics include:
- Precision: The proportion of relevant documents retrieved out of all documents retrieved.
- Evaluation Duration: The time taken to evaluate the search model.
- Retrieval Time: The time taken to retrieve documents from the vector store.
"""

from agent.tools.search import SearchModel
from eval.retriever.precision_eval_model import PrecisionEvalModel
from backend.agent.tools.retriever import Retriever

from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import time
from timeit import timeit

from argparse import ArgumentParser
import yaml

import os
import random
import tqdm

def parse_args():
    parser = ArgumentParser(description="Evaluate the search model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (YAML).")
    return parser.parse_args()

def evaluate_search_model(search_model, eval_model, prompt_samples):
    """
    Evaluate the search model using precision and retrieval time metrics.
    
    Args:
        search_model (SearchModel): The search model to evaluate.
        eval_model (PrecisionEvalModel): The model used for evaluating precision.
        prompt_samples (list): A list of prompt samples to evaluate.
    
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    metrics = {
        "precision_total": 0,
        "k": search_model.num_search,
        "retrieval_time": [],
        "precision": []
    }
    k = search_model.num_search

    for prompt in tqdm.tqdm(prompt_samples, desc="Evaluating prompts"):
        start_time = time.time()
        doc_list = search_model.invoke({"input": prompt})
        if type(doc_list) != list:
            doc_list = doc_list["content"]
        
        # log the retrieval time
        retrieval_time = time.time() - start_time
        metrics["retrieval_time"].append(retrieval_time)
        
        # evaluate the precision of the retrieved documents
        with ThreadPoolExecutor(max_workers=3) as executor:
            precision = 0
            futures = {executor.submit(eval_model.evaluate, prompt, doc.page_content): doc for doc in doc_list}
            
            for future in as_completed(futures):
                result = future.result()
                if "result" in result and result["result"] == 1:
                    precision += 1
                    metrics["precision_total"] += 1
                    
            # calculate the precision for this prompt
            precision /= len(doc_list)
            metrics["precision"].append(precision)
    # Average the metrics
    num_samples = len(prompt_samples)
    metrics[f"avg_precision@{k}"] = sum(metrics["precision"]) / len(metrics["precision"])
    metrics["average_retrieval_time"] = sum(metrics["retrieval_time"]) / num_samples

    return metrics

def save_metrics(metrics, metadata, model_kwargs, filename="eval_metrics.json"):
    """
    Save the evaluation metrics to a JSON file.
    
    Args:
        metrics (dict): The evaluation metrics to save.
        metadata (dict): Metadata about the model and evaluation.
        model_kwargs (dict): Additional model parameters used during evaluation.
        filename (str): The name of the file to save the metrics to.
    """
    for key in metadata:
        if key not in metrics:
            metrics[key] = metadata[key]
    for key in model_kwargs:
        if key not in metrics:
            metrics[key] = model_kwargs[key]
    
    path = os.path.dirname(os.path.abspath(__file__))
    file_path = path + "/logs/" + filename
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filename}")

def get_prompt_samples(num_samples, filename):
    """
    Load a randomly sampled number of prompt samples from a file.
    
    Args:
        num_samples (int): Number of samples to load.
    
    Returns:
        list: A list of prompt samples.
    """
    prompts = get_prompts_file(filename)
    if num_samples > len(prompts):
        raise ValueError(f"Requested {num_samples} samples, but only {len(prompts)} available.")
    return random.sample(prompts, num_samples)

def get_prompts_file(filename):
    """
    Load prompts from a file.
    
    Args:
        filename (str): Path to the file containing prompts.
    
    Returns:
        list: A list of prompts.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{current_dir}/{filename}", "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
        return prompts
    
def load_system_prompt(filename):
    """
    Load a system prompt from a file.
    
    Args:
        filename (str): Path to the file containing the system prompt.
    
    Returns:
        str: The system prompt.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{current_dir}/system_prompts/{filename}", "r") as f:
        return " ".join(f.readlines())
    

if __name__ == '__main__':
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{current_dir}/config/{args.config}", "r") as f:
        config = yaml.safe_load(f)
        metadata = config["metadata"]
        model_kwargs = config["model_kwargs"]
        search_params = config["search_params"]
        num_eval_samples = config["eval_params"]["num_samples"]
        sample_file = config["eval_params"]["sample_file"]
        
        # Baseline does not require a SearchModel
        if metadata["type"] == "LLM":
            model_kwargs["system_prompt"] = load_system_prompt(model_kwargs["system_prompt"])
        # initialize the search model
            search_model = SearchModel(version_name=metadata["name"], **model_kwargs)
        elif metadata["type"] == "crossencoder":
            # instead, search_model is the VectorStore
            model = config["model_kwargs"]["model"]
            docs_per_shard = search_params["docs_per_shard"]
            top_k = search_params["num_search"]
            
            search_model = Retriever(decomposition_model=None, encoder_model=model, 
                                     num_per_shard=docs_per_shard, top_k=top_k)

        precision_eval_model = PrecisionEvalModel(model_name=config["eval_params"]["eval_model"], temperature=0.05)
        # get the prompt samples
        prompt_samples = get_prompt_samples(num_eval_samples, sample_file)
        
        # evaluate the search model
        metrics = evaluate_search_model(search_model, precision_eval_model, prompt_samples)
        
        # save the evaluation metrics
        save_file = config["eval_params"]["save_file"]
        save_metrics(metrics, metadata, model_kwargs, filename=save_file)