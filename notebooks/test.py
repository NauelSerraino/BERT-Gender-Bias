import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer

# Generate some example data
rng = np.random.RandomState(1)
X = 5 * rng.rand(10000, 1)
continuous_y = np.random.rand(10000)  # Continuous values between 0 and 1

# Convert to binary using a threshold of 0.5
threshold = 0.5
y = (continuous_y > threshold).astype(int)

train_size = 100  # Adjust if needed

# Custom scoring function
def custom_scorer(estimator, X, y):
    accuracy = estimator.score(X, y)
    non_zero_coefs = np.sum(estimator.coef_ != 0)
    
    if np.isnan(accuracy) or np.isnan(non_zero_coefs):
        print("NaN detected in scoring!")
        return np.nan
    
    # Penalty calculation (adjust as needed)
    penalty = non_zero_coefs * 100000
    
    # Debug information
    print("Accuracy: ", accuracy)
    print("Number of non-zero coefficients: ", non_zero_coefs)
    print("Penalty: ", penalty)
    
    return accuracy - penalty

# Convert the custom scoring function to a scorer object
scorer = make_scorer(custom_scorer, greater_is_better=True)

# Define the model
logit = LogisticRegression(solver="liblinear", penalty="l1")

# Set up the parameter grid
param_grid = {'C': np.arange(1, 2, 0.5)}

# Perform grid search with custom scoring
grid_search = GridSearchCV(
    logit, 
    param_grid=param_grid, 
    scoring=scorer, 
    cv=5, 
    verbose=3, 
    error_score='raise'
)

# Fit the grid search
grid_search.fit(X[:train_size], y[:train_size])

# Extract the best model
best_logit = grid_search.best_estimator_

# Print results
print("Best C parameter: ", grid_search.best_params_['C'])
print("Custom scoring value: ", grid_search.best_score_)
print("Accuracy of the best model: ", best_logit.score(X, y))
print("Number of non-zero coefficients in the best model: ", np.sum(best_logit.coef_ != 0))
print("Number of zero coefficients in the best model: ", np.sum(best_logit.coef_ == 0))
print("Number of coefficients in the best model: ", len(best_logit.coef_[0]))
