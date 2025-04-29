import os
import logging
import pandas as pd 
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, path:str=""):
        logger.info(f"Initializing dataloader...")
        self.__path = path 
    
    def set_path(self, path:str):
        self.__path = path 
    
    def get_path(self) -> str: 
        return self.__path

    def load(self, max_reviews:int=None) -> pd.DataFrame:
        dataset_path = self.get_path()
        if not os.path.exists(dataset_path):
            logger.error(f"Wrong dataset path {dataset_path}")
            return 
        logger.info("Parsing dataset...")
        parsed_dataset = self._parse_dataset(dataset_path, max_reviews)
        logger.info("Creating dataframe...")
        data = pd.DataFrame(parsed_dataset)
        data.columns = self._polish_column_names(data.columns)
        return data

    def _parse_dataset(self, file_path:str, max_reviews:int=None) -> list[dict]:
        reviews = []
        review = {}
        with open(file_path, mode='r', encoding='iso-8859-1', errors='replace') as f:
            for line in tqdm(f, desc="Parsing line"):
                line = line.strip()
                if not line:
                    if review:
                        reviews.append(review)
                        review = {}
                        if max_reviews and len(reviews) >= max_reviews:
                            break
                else:
                    if ':' in line:
                        try:
                            key, value = line.split(":", 1)
                            review[key.strip()] = value.strip()
                        except Exception:
                            continue  
        if review and (not max_reviews or len(reviews) < max_reviews):
            reviews.append(review)
        return reviews
    
    def _polish_column_names(self, columns: list[str]) -> list[str]:
        return [col[col.find("/")+1:] for col in columns]