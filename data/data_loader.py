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
    """Loads and parses a structured text dataset into a pandas DataFrame."""
    def __init__(self, path:str=""):
        """Initializes the DataLoader with an optional file path."""
        logger.info(f"Initializing dataloader...")
        self.__path = path 
    
    def set_path(self, path:str):
        """Sets or updates the path to the dataset file."""
        self.__path = path 
    
    def get_path(self) -> str: 
        """Returns the path to the dataset file."""
        return self.__path

    def load(self, max_reviews:int=None) -> pd.DataFrame:
        """Loads the dataset from file, parses reviews, and returns a DataFrame. Return max_reviews number of rows or all rows."""
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
        """Parses the raw dataset file into a list of review dictionaries, optionally limiting the number of reviews."""
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
        """Cleans up column names by removing leading text up to and including the first '/' character."""
        return [col[col.find("/")+1:] for col in columns]