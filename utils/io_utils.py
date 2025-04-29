import os
import json
import joblib 
import pandas as pd
from sklearn.pipeline import Pipeline

def check_and_create_dir(file_path:str):
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

def save_metrics(metrics: dict[str,list], file_path:str):
  check_and_create_dir(file_path)
  pd.DataFrame(metrics).to_csv(file_path, index=False)

def save_reports(reports:list, file_path:str):
  check_and_create_dir(file_path)
  with open(file_path, "w", encoding="UTF-8") as f: 
    f.write(json.dumps(reports))

def save_pipeline(pipeline: Pipeline, file_path: str): 
  check_and_create_dir(file_path)
  joblib.dump(pipeline, file_path)

def load_pipeline(file_path: str) -> Pipeline:
  return joblib.load(file_path)