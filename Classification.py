import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from ucimlrepo import fetch_ucirepo


# Fetch the wine quality dataset using ucimlrepo library
def load_wine_quality_data():
    # Load wine quality dataset from UCI repository (ID: 186)
    wine_quality = fetch_ucirepo(id=186)

    # Extract features as a DataFrame and target as a Series
    data = pd.DataFrame(wine_quality.data.features, columns=wine_quality.data.feature_names)

    # Ensure targets are treated as a single column (series)
    target = wine_quality.data.targets
    if isinstance(target, pd.DataFrame):
        target = target.iloc[:, 0]  # Select the first column if it's a DataFrame

    target.name = "quality"  # Name the target for clarity

    X = data
    y = target
    return X, y


# Load data using the defined function
X, y = load_wine_quality_data()


# Define the function for Random Forest classification with cross-validation
def random_forest_experiment(n_estimators, min_samples_leaf, X, y):
    """
    Trains a RandomForestClassifier with specified parameters and performs 10-fold cross-validation.

    Parameters:
    - n_estimators: Number of trees in the forest.
    - min_samples_leaf: Minimum number of samples required to be at a leaf node.
    - X: Feature matrix (wine features).
    - y: Target vector (wine quality labels).

    Returns:
    - Mean cross-validation accuracy.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=2024)
    cv = KFold(n_splits=10, random_state=2024, shuffle=True)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()


# Function to return the best result based on hyperparameters
def get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y):
    """
    Finds the best hyperparameter combination based on cross-validation accuracy.

    Parameters:
    - n_estimators_list: List of different values for the number of trees.
    - min_samples_leaf_list: List of different values for min_samples_leaf.
    - X: Feature matrix (wine features).
    - y: Target vector (wine quality labels).

    Returns:
    - Best hyperparameter combination and corresponding accuracy.
    """
    best_accuracy = 0
    best_params = {'n_estimators': None, 'min_samples_leaf': None}

    for n_estimators in n_estimators_list:
        for min_samples_leaf in min_samples_leaf_list:
            accuracy = random_forest_experiment(n_estimators, min_samples_leaf, X, y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf}

    best_params['accuracy'] = best_accuracy
    return best_params


# TO DO: After running the above function, manually input your best result here
def manually_entered_best_params_and_accuracy():
    best_params = {'n_estimators': 100, 'min_samples_leaf': 10}  # Replace with actual best parameters
    best_accuracy = 0.72  # Replace with actual accuracy obtained
    return best_params, best_accuracy


# Main function
if __name__ == "__main__":
    # Experiment with different values for n_estimators and min_samples_leaf
    n_estimators_list = [10, 50, 100, 1000]
    min_samples_leaf_list = [1, 10, 50, 100]

    best_params = get_best_hyperparameters(n_estimators_list, min_samples_leaf_list, X, y)
    print(
        f"Best Hyperparameters: n_estimators = {best_params['n_estimators']}, min_samples_leaf = {best_params['min_samples_leaf']}")
    print(f"Best Accuracy: {best_params['accuracy']:.4f}")
