import os
import json
import joblib 
import pandas as pd
from sklearn.pipeline import Pipeline

def check_and_create_dir(file_path:str):
  """Create directory for the given file path if it doesn't exist."""
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

def save_metrics(metrics: dict[str,list], file_path:str):
  """Save evaluation metrics dictionary to a CSV file."""
  check_and_create_dir(file_path)
  pd.DataFrame(metrics).to_csv(file_path, index=False)

def save_reports(reports:list, file_path:str):
  """Save a list of evaluation reports to a JSON file."""
  check_and_create_dir(file_path)
  with open(file_path, "w", encoding="UTF-8") as f: 
    f.write(json.dumps(reports))

def save_pipeline(pipeline: Pipeline, file_path: str): 
  """Serialize and save a scikit-learn pipeline to a file."""
  check_and_create_dir(file_path)
  joblib.dump(pipeline, file_path)

def load_pipeline(file_path: str) -> Pipeline:
  """Load a scikit-learn pipeline from a file."""
  return joblib.load(file_path)