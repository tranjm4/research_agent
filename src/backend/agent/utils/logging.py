"""
File: src/backend/agent/utils/logging.py

This module provides logging utilities for the agent system.
"""
import json
import time
from pathlib import Path

VERBOSE = False

def log_stats(metadata, out_file, err_file):
    """
    Decorator to log the statistics of the model's call,
    including the number of tokens used and the time taken for the call.
    
    Args:
        metadata (dict): Metadata about the model call, including model type and version name.
        out_file (str): Path to the output file where the logged attributes will be saved.
        err_file (str): Path to the error file where any errors will be logged.
    Returns:
        A decorator that wraps the function to log the statistics.
    """
    def decorator(func):
        
        def wrapper(*args, **kwargs):
            try:
                start_time = time.perf_counter()
                output = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                execution_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                logged_attributes = {
                    "execution_time": execution_time,
                    "execution_date": execution_date,
                    "output": output.__repr__(),
                    "args": args.__repr__(),
                    "kwargs": kwargs.__repr__(),
                }
                for k, v in metadata.items():
                    logged_attributes[k] = v
                    
                # log the attributes to the output file
                out_file_path = Path(__file__).parent / out_file
                if VERBOSE:
                    print(f"Logging attributes to {out_file_path}")
                with open(out_file_path, "a") as f:
                    f.write(json.dumps(logged_attributes) + "\n")

                return output
            except FileNotFoundError as e:
                print(f"Error: {e}. Please check the file paths for out_file and err_file.")
                print(f"Received file paths: out_file={out_file}, err_file={err_file}")
            except Exception as e:
                # log the error to the error file
                err_file_path = Path(__file__).parent / err_file
                with open(err_file_path, "a") as f:
                    error_doc = {
                        "error": str(e),
                        "args": args,
                        "kwargs": kwargs,
                        "metadata": metadata,
                        "execution_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    }
                    f.write(json.dumps(error_doc) + "\n")
                print(f"An error occurred: {e}. Please check the error log at {err_file_path}")
                raise e
            
        return wrapper
    return decorator
