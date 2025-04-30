import os
import nltk
import random 
import logging
import pandas as pd 
from tqdm import tqdm
import nlpaug.augmenter.word as naw
from openai import OpenAI, AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

def build_augmentation_plan(data_df:pd.DataFrame, frequency_quantile:float, labels:list[str]) -> pd.DataFrame:
    """Creates a plan to augment underrepresented label combinations based on a quantile threshold."""
    label_pair_counts = data_df[labels].value_counts()
    frequency_threshold = label_pair_counts.quantile(frequency_quantile)
    augmentation_needs = (frequency_threshold - label_pair_counts).clip(lower=0).astype(int)
    augmentation_plan_df = augmentation_needs.reset_index()
    augmentation_plan_df.columns = labels + ['augmentations_needed']
    return augmentation_plan_df[augmentation_plan_df['augmentations_needed'] > 0]


class DataAugmenter:
    """Handles data augmentation using a variety of NLP augmenters for balancing datasets."""
    def __init__(self, data_df:pd.DataFrame, frequency_quantile:float, device:str="cpu"):
        """Initializes the augmenter with the dataset, quantile threshold, and device setting."""
        self.data_df = data_df
        self.data_df["source"] = "original"
        self.frequency_quantile = frequency_quantile
        self.augmentation_plan = None
        self.augmenters = None 
        self.augmenters_names = None
        self.augmenters_weights = None
        self.device = device
        self.initialize_augmenters()
    
    def initialize_augmenters(self):
        """Initializes a set of word and contextual augmenters, including OpenAI-based augmentation."""
        logger.info("Initializing augmenters")
        nltk.download('averaged_perceptron_tagger_eng')
        aug_wordnet = naw.SynonymAug(aug_src='wordnet')
        aug_delete = naw.RandomWordAug(action='delete')
        aug_swap = naw.RandomWordAug(action='swap')
        aug_sub = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action='substitute',
            aug_min=5,
            aug_max=20, 
            device=self.device
        )
        aug_insert = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action='insert',
            aug_min=2,
            aug_max=10,
            device=self.device
        )
        aug_gpt = OpenAIAugmenter()
        self.augmenters = {
            "wordnet_synonym": aug_wordnet,
            "random_delete": aug_delete,
            "random_swap": aug_swap,
            "contextual_substitute": aug_sub,
            "contextual_insert": aug_insert,
            "openai_gpt": aug_gpt
        }
        self.augmenters_names = list(self.augmenters.keys())
        self.augmenters_weights = [2/11] * 5 + [1/11]
    
    def assign_augmenters_weights(self, weights: list[float]):
       """Assigns custom weights to each augmenter for probabilistic selection during augmentation."""
       assert sum(weights) == 1.0, "Not a probability distribution sum(w) != 1"
       self.augmenters_weights = weights

    def prepare_augmentation_plan(self, label_columns: list[str]) -> pd.DataFrame:
       """Generates and stores an augmentation plan based on label frequency imbalance."""
       augmentation_plan = build_augmentation_plan(self.data_df, self.frequency_quantile, label_columns)
       self.augmentation_plan = augmentation_plan
       self.label_columns = label_columns
       return augmentation_plan
    
    def augment(self, output_prefix:str="", save_dir:str="data/augmented", save:bool=True) -> pd.DataFrame:
        """Applies augmentations based on the plan, optionally saving augmented and merged datasets."""
        if self.augmentation_plan is None or self.label_columns is None:
            raise ValueError("You must run prepare_augmentation_plan() first.")
        augmented_rows = []
        for _, row in self.augmentation_plan.iterrows():
            conditions = (self.data_df[self.label_columns[0]] == row[self.label_columns[0]])
            for col in self.label_columns[1:]:
                conditions &= (self.data_df[col] == row[col])
            matching_rows = self.data_df[conditions]
        
            n = int(row['augmentations_needed'])
            sample_rows = matching_rows.sample(n=n, replace=(n > len(matching_rows)), random_state=42)
            for _, sample_row in tqdm(sample_rows.iterrows(), desc='Augmenting rows', total=len(sample_rows)) :
                original_text = sample_row['text']
                original_summary = sample_row['summary']
                augmented_texts = self.augment_text(original_text)
                augmented_summaries = self.augment_text(original_summary)
                for (aug_text_name, new_text), (aug_summary_name, new_summary) in zip(augmented_texts, augmented_summaries):
                    new_row = sample_row.copy()
                    new_row['text'] = new_text
                    new_row['summary'] = new_summary
                    new_row['source'] = 'augmentation'
                    new_row['augmentation_used_text'] = aug_text_name
                    new_row['augmentation_used_summary'] = aug_summary_name
                    augmented_rows.append(new_row)
        logger.info("Done augmenting")
        augmented_df = pd.DataFrame(augmented_rows, columns=self.data_df.columns.tolist() + ['augmentation_used_text', 'augmentation_used_summary'])
        if save: 
            logger.info("Merging augmented and original data...")
            os.makedirs(save_dir, exist_ok=True)
            merged_df = pd.concat([self.data_df, augmented_df], ignore_index=True).drop_duplicates()
            path_aug = os.path.join(save_dir, f"{output_prefix}_augmented.csv")
            path_merged = os.path.join(save_dir, f"{output_prefix}_augmented_merged.csv")
            augmented_df.to_csv(path_aug, index=False)
            merged_df.to_csv(path_merged, index=False)
            logger.info(f"Saved augmented data to {path_aug}")
            logger.info(f"Saved merged data to {path_merged}")
        return augmented_df

    def augment_text(self, text:str, num_aug:int=1) -> list[(str, str)]:
        """Applies a randomly chosen augmenter to a given text and returns augmented results."""
        results = []
        augmenter_names = random.choices(self.augmenters_names, self.augmenters_weights, k=num_aug)
        for aug_name in augmenter_names:
            try:
                aug_text = self.augmenters[aug_name].augment(text)
                if isinstance(aug_text, list):
                    aug_text = aug_text[0]
                results.append((aug_name, aug_text))
            except Exception as e:
                results.append((None, text)) # Will need to drop duplicates
        return results           


class OpenAIAugmenter:
    """Uses OpenAI's API to generate paraphrased text based on a configurable prompt."""
    def __init__(self):
        """Initializes OpenAI client with a default prompt for food hazard-related text augmentation."""
        logger.info("Initiliazing OpenAI Augmenter with default prompt. Make sure to provide api key as env var!")
        api_key = os.getenv("OPENAI_API_KEY", "SECRET-API-KEY") 
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.augment_prompt = (
        "Paraphrase the following text related to fine food to generate a new training example. "
        "Maintain the original meaning and topic, but vary the wording naturally. Return the final text only:\n\n\"{}\""
        )

    def set_augment_prompt(self, prompt:str):
        """Sets a custom prompt for text augmentation using OpenAI."""
        self.augment_prompt = prompt

    def get_augment_prompt(self) -> str:
        """Returns the current augmentation prompt used for OpenAI generation."""
        return self.augment_prompt

    def get_response(self, prompt:str) -> str:
        """Sends the prompt to OpenAI's API and returns the generated text response."""
        response = self.client.responses.create(
            model="gpt-4o",
            input=prompt
        )
        return response.output[0].content[0].text

    def augment(self, text: str) -> str:
        """Formats the prompt with input text and returns the paraphrased result from OpenAI."""
        prompt = self.get_augment_prompt().format(text)
        return self.get_response(prompt)