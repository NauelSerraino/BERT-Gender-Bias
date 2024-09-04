from gensim.models import KeyedVectors
import pandas as pd
import os
import numpy as np
from transformers import BertTokenizer, BertModel

import sys
sys.path.append('/home/nauel/bert_gender_bias')
from pipelines.bert_gender_bias.train import FeatureSelectionPipeline
from pipelines.bert_gender_bias.etl import PreProcessPipeline, BertEmbeddingsPipeline

from utils.paths import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
import torch


class GenderBiasPipeline():
    def __init__(self):
        self.preprocess = PreProcessPipeline()
        self.bert_embeddings = BertEmbeddingsPipeline()
        self.feat_selection = FeatureSelectionPipeline()
        
    def run(self):
        # self.preprocess.run()
        # self.bert_embeddings.run()
        X, y = self._get_x_y('01_gender_binary_words.pkl')
        self.feat_selection.fit(X, y)
        results = self.feat_selection.get_results()
        print(f"C: {results['Best Parameters']['feature_selection__estimator__C']}")
        print(f"Accuracy: {results['Best Accuracy']}")
        print(f"Number of Selected Features: {results['Number of Selected Features']}")
        
    def _get_x_y(self, file_name):
        df = pd.read_pickle(os.path.join(INTERIM_DATA_DIR, file_name))
        X = df['bert_token'].values
        y = df['gender_binary'].values
        return X, y
    
        

if __name__ == '__main__':
    pipeline = GenderBiasPipeline()
    pipeline.run()