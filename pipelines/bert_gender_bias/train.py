import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

class FeatureSelectionPipeline:
    def __init__(self, n_pca_components=2, cv=10, scoring='accuracy', verbose=3):
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
        X = np.array([np.array(xi) for xi in X])
            
        print(f"shape of X: {X.shape}")
        print(f"shape of y: {y.shape}")
        print(f"shape of x[0]: {X[0].shape}")
            
        self.pipeline = Pipeline([
            ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'))),
            ('pca', PCA(n_components=self.n_pca_components)),
            ('svm', SVC())
        ])

        param_grid = {
            'feature_selection__estimator__C': np.logspace(-2, 2, 100),
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

    def get_results(self):
        return {
            "Best Parameters": self.best_params_,
            "Best Accuracy": self.best_score_,
            "Number of Selected Features": self.num_selected_features
        }
        

# Example usage with some data (replace with your own data)
if __name__ == "__main__":
    # X, y = make_classification(n_samples=500, n_features=700, n_informative=10, n_classes=2, random_state=42)
    X = np.random.rand(515, 768)  # Replace with actual data
    y = np.random.randint(0, 2, size=515) 

    # Initialize the custom pipeline class
    pipeline = FeatureSelectionPipeline()

    # Fit the pipeline to the data
    pipeline.fit(X, y)

    # Retrieve and print the results
    results = pipeline.get_results()
    print(results)
