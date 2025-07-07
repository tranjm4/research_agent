"""
File: src/agent/eval/eval_metrics.py

This module provides functionality to evaluate the performance of the agent
using various metrics. It includes methods to compute precision, recall, and F1 score,
as well as to evaluate the agent's performance based on the number of relevant documents retrieved.
"""

from datetime import datetime
from argparse import ArgumentParser
from langchain_ollama import ChatOllama




def compute_precision(retrieved: int, relevant: int) -> float:
    """
    Compute the precision of the agent's retrieval.

    Args:
        retrieved (int): The number of documents retrieved by the agent.
        relevant (int): The number of relevant documents.

    Returns:
        float: The precision score.
    """
    if retrieved == 0:
        return 0.0
    return relevant / retrieved

def compute_recall(retrieved: int, relevant: int) -> float:
    """
    Compute the recall of the agent's retrieval.

    Args:
        retrieved (int): The number of documents retrieved by the agent.
        relevant (int): The number of relevant documents.

    Returns:
        float: The recall score.
    """
    if relevant == 0:
        return 0.0
    return retrieved / relevant

def compute_f1_score(precision: float, recall: float) -> float:
    """
    Compute the F1 score based on precision and recall.

    Args:
        precision (float): The precision score.
        recall (float): The recall score.

    Returns:
        float: The F1 score.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_faithfulness(context: str, response: str) -> float:
    """
    Compute the faithfulness of the agent's retrieval.

    Args:
        retrieved (int): The number of documents retrieved by the agent.
        relevant (int): The number of relevant documents.

    Returns:
        float: The faithfulness score.
    """

def compute_relevance(retrieved: int, relevant: int) -> float:
    """
    Compute the relevance of the agent's retrieval.

    Args:
        retrieved (int): The number of documents retrieved by the agent.
        relevant (int): The number of relevant documents.

    Returns:
        float: The relevance score.
    """
    if relevant == 0:
        return 0.0
    return retrieved / relevant

def evaluate_agent_performance(model: str, retriever, generator) -> dict:
    """
    Evaluate the agent's performance based on the number of relevant documents retrieved.

    Args:
        retrieved (int): The number of documents retrieved by the agent.
        relevant (int): The number of relevant documents.

    Returns:
        dict: A dictionary containing precision, recall, and F1 score.
    """
    precision = compute_precision(retrieved, relevant)
    recall = compute_recall(retrieved, relevant)
    f1_score = compute_f1_score(precision, recall)
    faithfulness = compute_faithfulness(context, response)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Evaluating agent performance for model '{model}':")
    print(f"  Retrieved: {retrieved}, Relevant: {relevant}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    print(f"  Faithfulness: {retrieved / relevant:.4f}, Relevance: {retrieved / relevant:.4f}")
    
    with open("eval_logs.txt", "a") as log_file:
        log_file.write(f"[{current_time}] Model: {model}, Retrieved: {retrieved}, Relevant: {relevant}, "
                       f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n")
        
        
MODELS = {
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.1:8b",
    "llama3.1:70b",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
}
        

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate agent performance metrics.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model used by the agent.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for the model.  Defaults to 0.1.")
    parser.add_argument("--samples_file", type=str, required=True, help="Path to the file containing the number of retrieved and relevant documents.")

    args = parser.parse_args()

    evaluate_agent_performance(args.model_name, args.retrieved_docs, args.relevant_docs)