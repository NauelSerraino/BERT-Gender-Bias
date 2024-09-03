from gensim.models import KeyedVectors
import pandas as pd
import os
import numpy as np
from transformers import BertTokenizer, BertModel

import sys
sys.path.append('/home/nauel/bert_gender_bias')
from pipelines.bert_gender_bias.etl import PreProcessPipeline, BertEmbeddingsPipeline
from utils.paths import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
import torch


class GenderBiasPipeline():
    def __init__(self):
        self.preprocess = PreProcessPipeline()
        self.bert_embeddings = BertEmbeddingsPipeline()
        
    def run(self):
        self.preprocess.run()
        self.bert_embeddings.run()
        

if __name__ == '__main__':
    pipeline = GenderBiasPipeline()
    
    pipeline.run()