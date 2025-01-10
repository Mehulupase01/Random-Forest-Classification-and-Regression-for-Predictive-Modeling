# Random Forest Classification and Regression for Predictive Modeling
 This project implements Random Forest models for both classification and regression tasks. The classification task uses the Wine Quality dataset, while the regression task uses the Bike Sharing dataset. Hyperparameters are tuned using 10-fold cross-validation to identify the best settings for both tasks

# Random Forests for Classification and Regression

This project implements **Random Forest** models for **classification** and **regression** tasks. The classification task uses the **Wine Quality dataset** to predict wine quality, and the regression task uses the **Bike Sharing dataset** to predict the number of bike rentals. Hyperparameters are tuned using 10-fold cross-validation.

## Overview

The project consists of two main tasks:

1. **Random Forest Classification**: Using the **Wine Quality dataset** to predict wine quality based on various features. The task involves training a Random Forest classifier, performing cross-validation, and tuning hyperparameters like the number of trees (`n_estimators`) and the minimum samples per leaf (`min_samples_leaf`).
   
2. **Random Forest Regression**: Using the **Bike Sharing dataset** to predict the number of bike rentals. This task also involves training a Random Forest regressor, performing cross-validation, and tuning hyperparameters.

### Key Tasks:
1. **Classification Task**: 
   - Train a **RandomForestClassifier** on the Wine Quality dataset.
   - Perform 10-fold cross-validation.
   - Tune `n_estimators` and `min_samples_leaf` to find the best model.
   
2. **Regression Task**:
   - Train a **RandomForestRegressor** on the Bike Sharing dataset.
   - Perform 10-fold cross-validation.
   - Tune `n_estimators` and `min_samples_leaf` to find the best model.

## Code Structure

The project consists of two Python files, each corresponding to one of the tasks:
1. **`Task_Classification_4017633.py`**: Implements the classification task for predicting wine quality.
2. **`Task_Regression_4017633.py`**: Implements the regression task for predicting bike rentals.

### Libraries Used:
- **pandas**: For data manipulation and preprocessing.
- **scikit-learn**: For building Random Forest models, cross-validation, and hyperparameter tuning.
- **ucimlrepo**: For fetching datasets from the UCI repository.

## Output

- **Classification Output**: For the classification task, the program prints the best hyperparameters (`n_estimators` and `min_samples_leaf`) and the cross-validation accuracy.
  
Example output for classification:
```
Best Hyperparameters: n_estimators = 100, min_samples_leaf = 10 Best Accuracy: 0.72
```

- **Regression Output**: For the regression task, the program prints the best hyperparameters (`n_estimators` and `min_samples_leaf`) and the cross-validation negative mean squared error (MSE).

Example output for regression:
```
Best Hyperparameters: n_estimators = 1000, min_samples_leaf = 1 Best Negative MSE: -418641.1448
```


## Results & Discussion

- **Classification Task**: The Random Forest classifier was successfully trained on the Wine Quality dataset, with hyperparameter tuning revealing the best model settings. The classifier achieved a satisfactory accuracy of around 72%.
  
- **Regression Task**: The Random Forest regressor was trained on the Bike Sharing dataset, where the model achieved a negative MSE of -418,641. The hyperparameter tuning resulted in the best settings for accurate bike rental prediction.

## Conclusion

This project demonstrates the application of **Random Forests** for both **classification** and **regression** tasks. By tuning the hyperparameters using **10-fold cross-validation**, the models performed well in predicting wine quality and bike rentals. Random Forests proved to be an effective algorithm for both tasks, providing robust predictions on the given datasets.

## References

1. **Wine Quality Dataset**: UCI Machine Learning Repository (ID: 186).
2. **Bike Sharing Dataset**: UCI Machine Learning Repository (ID: 275).
3. **Random Forests**: Breiman, L. (2001). "Random Forests." Machine Learning.

