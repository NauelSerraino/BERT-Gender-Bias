from gensim.models import KeyedVectors
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

import sys
sys.path.append('/home/nauel/bert_gender_bias')
from pipelines.bert_gender_bias.train import FeatureSelectionPipeline
from pipelines.bert_gender_bias.etl import PreProcessPipeline, BertEmbeddingsPipeline

from utils.paths import EXTERNAL_DATA_DIR, FINAL_DATA_DIR, INTERIM_DATA_DIR, FIGURES_DIR
import torch


class GenderBiasPipeline():
    def __init__(self):
        self.preprocess = PreProcessPipeline()
        self.bert_embeddings = BertEmbeddingsPipeline()
        self.feat_selection = FeatureSelectionPipeline(cv=5)
        
    def run(self):
        # self.preprocess.run()
        # self.bert_embeddings.run()
        self._get_x_y('01_gender_binary_words.pkl')
        self.feat_selection.fit(self.X, self.y)
        results = self.feat_selection.get_results()
        
        print(f"C logit: {results['Best Parameters']['feature_selection__estimator__C']}")
        print(f"SVM C: {results['Best Parameters']['svm__C']}")
        print(f"Accuracy: {results['Best Accuracy']}")
        print(f"Number of Selected Features: {results['Number of Selected Features']}")
        
        pca_df = self.feat_selection.export_pca_components(self.y)
        cv_results_df = self.feat_selection.export_accuracy_cv_results()
        
        self._plot_cv_results(cv_results_df)
        self._plot_pca_with_target(pca_df)
        self.feat_selection.plot_svm_decision_boundary(self.y)
        
    def _get_x_y(self, file_name):
        self.df = pd.read_pickle(os.path.join(INTERIM_DATA_DIR, file_name))
        self.X = self.df['bert_token'].values
        self.y = self.df['gender_binary'].values
        
    
    def _plot_cv_results(self, cv_results_df):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.errorbar(cv_results_df['param_feature_selection__estimator__C'], 
                    cv_results_df['mean_test_score'], 
                    yerr=cv_results_df['std_test_score'], 
                    fmt='o', capsize=5, color='blue', label='Mean Accuracy')
        
        ax1.set_xlabel('C Parameter')
        ax1.set_ylabel('Mean Accuracy', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title('Cross Validation Results')
        
        fig.tight_layout()
        ax1.grid(True)
        plt.savefig(os.path.join(FIGURES_DIR, 'cv_result.png'))
        plt.show()
        
    def _plot_pca_with_target(self, pca_df):
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'pink']
        alpha = 0.8
        for i, color in enumerate(colors):
            plt.scatter(
                pca_df.loc[pca_df['target'] == i, 'PC1'], 
                pca_df.loc[pca_df['target'] == i, 'PC2'], 
                color=color, 
                alpha=alpha
                )
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA with Gender Target')
        plt.legend(['0=Male', '1=Female'])
        plt.grid(True, which='both', linestyle='--', lw=0.5)
        plt.savefig(os.path.join(FIGURES_DIR, 'pca_with_target.png'))
        plt.show()



if __name__ == '__main__':
    pipeline = GenderBiasPipeline()
    pipeline.run()
    predictions_df = pipeline.feat_selection.predict(pipeline.df)
    predictions_df.to_csv(os.path.join(FINAL_DATA_DIR, '00_predictions.csv'), index=False)
    
    occupations_df = pd.read_pickle(os.path.join(INTERIM_DATA_DIR, '01_occupations.pkl'))
    predictions_df_occupations = pipeline.feat_selection.predict(occupations_df)
    predictions_df_occupations.to_csv(os.path.join(FINAL_DATA_DIR, '01_predictions_occupations.csv'), index=False)