import numpy as np
import pandas as pd
import os
import sys
import torch
from transformers import BertTokenizer, BertModel

sys.path.append('/home/nauel/bert_gender_bias')
from pipelines.utils.paths import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR


class PreProcessPipeline:
    """
    Preprocess the raw data of male/female words to be used in the model.
    """
    def __init__(self):
        self.external_data_dir = EXTERNAL_DATA_DIR
        self.interim_data_dir = INTERIM_DATA_DIR
        self.gender_binary_path = os.path.join(self.interim_data_dir, '00_gender_binary_words.csv')
        self.occupations_path = os.path.join(self.interim_data_dir, '00_occupations.csv')

    def run(self):
        self._load_raw()
        self._save_words()
        
    def _load_raw(self):
        self.__male_female_df()
        self.__occupations_df()

    def _save_words(self):
        self.male_female_df.to_csv(self.gender_binary_path, index=False, sep="|")
        self.occupations_df.to_csv(self.occupations_path, index=False, sep="|")

    def __male_female_df(self):
        male_words = pd.read_csv(os.path.join(self.external_data_dir, "male.txt"), header=None)
        male_words.columns = ['word']
        male_words['gender_binary'] = 0

        female_words = pd.read_csv(os.path.join(self.external_data_dir, "female.txt"), header=None)
        female_words.columns = ['word']
        female_words['gender_binary'] = 1

        self.male_female_df = pd.concat([male_words, female_words])
        self.male_female_df = self.male_female_df.sample(frac=1).reset_index(drop=True)
        
    def __occupations_df(self):
        self.occupations_df = pd.read_csv(os.path.join(self.external_data_dir, "occupations.csv"))
        self.occupations_df.columns = ['word']
        self.occupations_df
        
        

class BertEmbeddingsPipeline:
    """
    Use BERT to generate embeddings for the words, given a DataFrame with the words.
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")
        
        self.interim_data_dir = INTERIM_DATA_DIR
        self.gender_binary_path_input = os.path.join(self.interim_data_dir, '00_gender_binary_words.csv')
        self.occupations_path_input = os.path.join(self.interim_data_dir, '00_occupations.csv')
        self.gender_binary_path_output = os.path.join(self.interim_data_dir, '01_gender_binary_words.csv')
        self.occupations_path_output = os.path.join(self.interim_data_dir, '01_occupations.csv')
        
        self.gender_binary = None
        self.occupations = None
        
    def run(self):
        self._load_data()
        
        self.gender_binary = self._generate_embeddings(self.gender_binary)
        self.occupations = self._generate_embeddings(self.occupations)
        
        self._save_embeddings(self.gender_binary, self.gender_binary_path_output)
        self._save_embeddings(self.occupations, self.occupations_path_output)
        
    def _load_data(self):
        self.gender_binary = pd.read_csv(self.gender_binary_path_input, sep="|")
        self.occupations = pd.read_csv(self.occupations_path_input, sep="|")
        
    def _generate_embeddings(self, df):
        text = df.word.values.tolist()
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            output = self.model(**encoded_input)
            
        embeddings = output.last_hidden_state
        word_embeddings = embeddings.mean(dim=1).numpy()
        
        df['bert_token'] = list(word_embeddings)
        return df

    def _save_embeddings(self, df, path):
        df.to_csv(path, index=False, sep="|")