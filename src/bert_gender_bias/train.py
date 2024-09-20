from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV


from utils.paths import FIGURES_DIR, MODELS_DIR

class FeatureSelectionPipeline:
    def __init__(self, n_pca_components=2, cv=5, scoring='accuracy', verbose=3):
        self.n_pca_components = n_pca_components
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.pipeline = None
        self.grid_search = None
        self.best_pipeline = None
        self.best_params_ = None
        self.best_score_ = None
        self.num_selected_features = None

    def fit(self, X, y):       
        X = np.array([np.array(xi) for xi in X]) #TODO: this makes the X in the proper format, but you should be able to save the pkl directly with the proper format 
            
        self.pipeline = Pipeline([
            ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'))),
            ('pca', PCA(n_components=self.n_pca_components)),
            ('svm', SVC(kernel='linear', probability=True))
        ])

        param_grid = {
            'feature_selection__estimator__C': np.arange(0.1, 0.5, 0.025),
            'svm__C': np.arange(0.1, 0.5, 0.025)
        }

        self.grid_search = GridSearchCV(
            self.pipeline, 
            param_grid, 
            cv=self.cv, 
            scoring=self.scoring, 
            verbose=self.verbose
            )
        self.grid_search.fit(X, y)
        
        self.best_pipeline = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_
        self.best_score_ = self.grid_search.best_score_
        selected_features_mask = self.best_pipeline.named_steps['feature_selection'].get_support()
        self.num_selected_features = selected_features_mask.sum()
        
        self._save_model()
        
        pre_transformed_X = self.best_pipeline.named_steps['feature_selection'].transform(X)
        self.principal_components = self.best_pipeline.named_steps['pca'].transform(pre_transformed_X)
        
    def get_results(self):
        return {
            "Best Parameters": self.best_params_,
            "Best Accuracy": self.best_score_,
            "Number of Selected Features": self.num_selected_features
        }
    
    def export_accuracy_cv_results(self):
        cv_results = self.grid_search.cv_results_
        cv_results_df = pd.DataFrame(cv_results)
        return cv_results_df
        
    def export_pca_components(self, y):
        pcs = self.principal_components
        pcs_df = pd.DataFrame(pcs)
        pcs_df.columns = ['PC1', 'PC2']
        pcs_df['target'] = y
        return pcs_df
    
    def plot_svm_decision_boundary(self, y):
        pcs = self.principal_components
        h = .02  # step size in the mesh

        x_min, x_max = pcs[:, 0].min() - 1, pcs[:, 0].max() + 1
        y_min, y_max = pcs[:, 1].min() - 1, pcs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = self.best_pipeline.named_steps['svm'].predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        colors = ['blue', 'pink']
        alpha = 0.8
        for i, color in enumerate(colors):
            plt.scatter(
                pcs[y == i, 0], 
                pcs[y == i, 1], 
                color=color, 
                alpha=alpha
            )
        
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('SVM Decision Boundary with PCA Components')
        plt.legend(['0=Male', '1=Female'])
        plt.grid(True)
        plt.savefig(os.path.join(FIGURES_DIR, 'svm_decision_boundary.png'))
        plt.show()
        
    def predict(self, df):
        """
        Returns the predictions of the best pipeline on the given df.
        """
        bert_embeddings = np.array(df['bert_token'].tolist())
        predictions = self.best_pipeline.predict(bert_embeddings)
        probabilities = self.best_pipeline.predict_proba(bert_embeddings)
        
        prob_male = probabilities[:, 0]
        prob_female = probabilities[:, 1]
        
        data = {
        "Words": df['word'],
        "Predictions": predictions,
        "Probability_Male": prob_male,
        "Probability_Female": prob_female
        }
        
        if 'gender_binary' in df.columns:
            data['Target'] = df['gender_binary']
        
        pred_df = pd.DataFrame(data)
        
        return pred_df

    def _save_model(self):
        
        joblib.dump(self.best_pipeline, os.path.join(MODELS_DIR, 'best_pipeline.pkl'))
        joblib.dump({
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'num_selected_features': self.num_selected_features,
            'accuracy_cv_results': self.grid_search.cv_results_,
            'best_accuracy_cv_results': self.grid_search.best_score_,
            'C_svc': self.best_params_['svm__C'],
            'C_logit': self.best_params_['feature_selection__estimator__C']
        }, os.path.join(MODELS_DIR, 'best_pipeline_metadata.pkl'))
        
# Example usage with some data (replace with your own data)
# if __name__ == "__main__":
#     # X, y = make_classification(n_samples=500, n_features=700, n_informative=10, n_classes=2, random_state=42)
#     X = np.random.rand(515, 768)  # Replace with actual data
#     y = np.random.randint(0, 2, size=515) 

#     # Initialize the custom pipeline class
#     pipeline = FeatureSelectionPipeline()

#     # Fit the pipeline to the data
#     pipeline.fit(X, y)

#     # Retrieve and print the results
#     results = pipeline.get_results()
#     print(results)
