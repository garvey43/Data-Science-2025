# Lecture 14.9: Random Forest Classification

## Key Learning Objectives
- Understand Random Forest as an extension of bagging
- Learn random feature selection and its benefits
- Master hyperparameter tuning for Random Forest
- Implement scalable Random Forest with RAPIDS

## Core Concepts

### From Bagging to Random Forest

#### Bagging Review
- **Bootstrap sampling**: Random sampling with replacement
- **Base estimators**: Any model (typically decision trees)
- **Aggregation**: Majority vote for classification

#### Random Forest Innovation
- **Additional randomness**: Random feature selection at each split
- **Decorrelates trees**: Reduces correlation between individual trees
- **Better generalization**: Less prone to overfitting

### Random Feature Selection

#### Feature Subset Selection
```python
import numpy as np

def select_random_features(X, max_features):
    """
    Select random subset of features
    """
    n_features = X.shape[1]
    
    if max_features == 'sqrt':
        n_selected = int(np.sqrt(n_features))
    elif max_features == 'log2':
        n_selected = int(np.log2(n_features))
    elif isinstance(max_features, float):
        n_selected = int(max_features * n_features)
    else:
        n_selected = max_features
    
    # Randomly select feature indices
    feature_indices = np.random.choice(n_features, n_selected, replace=False)
    return feature_indices

# Example
X = np.random.randn(100, 10)  # 100 samples, 10 features
selected_features = select_random_features(X, max_features='sqrt')
print(f"Selected {len(selected_features)} features: {selected_features}")
```

## Random Forest Algorithm

### Training Process
1. **Bootstrap sampling**: Create multiple bootstrap samples from training data
2. **Feature randomization**: For each tree, randomly select subset of features
3. **Tree construction**: Build decision tree using only selected features
4. **Repeat**: Create multiple trees with different feature subsets

### Prediction Process
- **Individual predictions**: Each tree makes a prediction
- **Majority voting**: Classification based on most frequent prediction
- **Probability estimation**: Average predicted probabilities across trees

## Implementation in Scikit-Learn

### Basic Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create Random Forest classifier
rf = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    criterion='gini',        # Splitting criterion
    max_depth=None,          # Maximum tree depth
    min_samples_split=2,     # Minimum samples to split
    min_samples_leaf=1,      # Minimum samples per leaf
    max_features='sqrt',     # Features to consider at each split
    bootstrap=True,          # Use bootstrap sampling
    oob_score=True,          # Enable OOB scoring
    random_state=42,
    n_jobs=-1                # Use all available cores
)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"OOB Score: {rf.oob_score_:.3f}")
print(classification_report(y_test, y_pred))
```

### Key Hyperparameters

#### Number of Trees
```python
# Test different numbers of trees
n_estimators_range = [10, 50, 100, 200, 500]
train_scores = []
oob_scores = []

for n_est in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n_est, random_state=42, oob_score=True)
    rf_temp.fit(X_train, y_train)
    
    train_pred = rf_temp.predict(X_train)
    train_scores.append(accuracy_score(y_train, train_pred))
    oob_scores.append(rf_temp.oob_score_)

# Plot convergence
plt.plot(n_estimators_range, train_scores, label='Training Accuracy')
plt.plot(n_estimators_range, oob_scores, label='OOB Score')
plt.xlabel('Number of Trees')
plt.ylabel('Score')
plt.legend()
plt.show()
```

#### Maximum Features
```python
# Test different max_features values
max_features_options = ['sqrt', 'log2', None, 0.5]
scores = []

for max_feat in max_features_options:
    rf_temp = RandomForestClassifier(
        n_estimators=100, 
        max_features=max_feat, 
        random_state=42, 
        oob_score=True
    )
    rf_temp.fit(X_train, y_train)
    scores.append(rf_temp.oob_score_)

# Display results
for feat, score in zip(max_features_options, scores):
    print(f"max_features={feat}: OOB Score = {score:.3f}")
```

## Feature Importance

### Gini Importance (Mean Decrease in Impurity)
```python
# Feature importance from scikit-learn
feature_importance = rf.feature_importances_
feature_names = X.columns

# Sort and display
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.xlabel('Feature Importance')
plt.title('Top 10 Important Features')
plt.gca().invert_yaxis()
plt.show()
```

### Permutation Importance
```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(
    rf, X_test, y_test, 
    n_repeats=10, 
    random_state=42
)

# Display results
perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print(perm_df.head(10))
```

## Out-of-Bag (OOB) Error

### OOB Estimation
- **Concept**: Use trees that didn't see a sample during training to validate it
- **Advantage**: Built-in cross-validation without additional data split
- **Calculation**: Approximately 37% of samples are OOB for each tree

### OOB vs Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# OOB score (built-in)
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)
oob_score = rf_oob.oob_score_

# 5-fold CV score
cv_scores = cross_val_score(rf_oob, X_train, y_train, cv=5)
cv_score = cv_scores.mean()

print(f"OOB Score: {oob_score:.3f}")
print(f"CV Score: {cv_score:.3f} (+/- {cv_scores.std()*2:.3f})")
```

## Handling Class Imbalance

### Class Weighting
```python
# Automatic class balancing
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Automatic weighting
    random_state=42
)

# Manual class weights
class_weights = {0: 1, 1: 5}  # Higher weight for minority class
rf_manual = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weights,
    random_state=42
)
```

### Balanced Random Forest
```python
from imblearn.ensemble import BalancedRandomForestClassifier

# Balanced Random Forest
brf = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42,
    sampling_strategy='auto',  # Balance classes
    replacement=True
)

brf.fit(X_train, y_train)
```

## Random Forest with RAPIDS

### GPU-Accelerated Random Forest
```python
import cudf
from cuml.ensemble import RandomForestClassifier as CumlRandomForest
from cuml.metrics import accuracy_score

# Convert to GPU dataframes
X_gpu = cudf.DataFrame.from_pandas(X_train)
y_gpu = cudf.Series(y_train)

# GPU Random Forest
gpu_rf = CumlRandomForest(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42
)

gpu_rf.fit(X_gpu, y_gpu)

# Predictions
y_pred_gpu = gpu_rf.predict(X_gpu)
y_prob_gpu = gpu_rf.predict_proba(X_gpu)

# Evaluate
accuracy = accuracy_score(y_gpu, y_pred_gpu)
print(f"GPU Accuracy: {accuracy:.3f}")
```

### Performance Comparison
```python
import time

# CPU Random Forest
start_time = time.time()
cpu_rf = RandomForestClassifier(n_estimators=100, random_state=42)
cpu_rf.fit(X_train, y_train)
cpu_time = time.time() - start_time

# GPU Random Forest
start_time = time.time()
gpu_rf.fit(X_gpu, y_gpu)
gpu_time = time.time() - start_time

print(f"CPU time: {cpu_time:.2f}s")
print(f"GPU time: {gpu_time:.2f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

## Hyperparameter Tuning

### Grid Search for Random Forest
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, oob_score=True),
    param_grid,
    cv=3,  # 3-fold CV for speed
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Best OOB score: {grid_search.best_estimator_.oob_score_:.3f}")
```

### Randomized Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")
```

## Advantages and Limitations

### Advantages
- **High accuracy**: Often best out-of-the-box performance
- **Robust to overfitting**: Ensemble nature prevents overfitting
- **Handles mixed data**: Works with numerical and categorical features
- **Feature importance**: Built-in feature selection
- **Parallel training**: Trees can be trained independently
- **OOB validation**: Built-in cross-validation

### Limitations
- **Black box**: Hard to interpret individual predictions
- **Memory intensive**: Stores multiple trees
- **Slow prediction**: Must evaluate all trees
- **Not suitable for streaming**: Requires batch training
- **Hyperparameter sensitive**: Many parameters to tune

## Practical Implementation

### Complete Random Forest Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest with OOB scoring
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Model evaluation
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("=== Model Performance ===")
print(f"OOB Score: {rf.oob_score_:.3f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Test AUC: {roc_auc_score(y_test, y_prob):.3f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 10 Important Features ===")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Learning curve analysis
train_scores = []
oob_scores = []
n_estimators_range = [10, 25, 50, 100, 200]

for n_est in n_estimators_range:
    rf_temp = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=10,
        random_state=42,
        oob_score=True
    )
    rf_temp.fit(X_train, y_train)
    
    train_pred = rf_temp.predict(X_train)
    train_scores.append(accuracy_score(y_train, train_pred))
    oob_scores.append(rf_temp.oob_score_)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'b-', label='Training Accuracy')
plt.plot(n_estimators_range, oob_scores, 'r-', label='OOB Score')
plt.xlabel('Number of Trees')
plt.ylabel('Score')
plt.title('Random Forest Learning Curves')
plt.legend()
plt.grid(True)
plt.show()

# Cross-validation for robustness check
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Compare with single decision tree
from sklearn.tree import DecisionTreeClassifier

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_pred = single_tree.predict(X_test)
single_accuracy = accuracy_score(y_test, single_pred)

print(f"\nComparison:")
print(f"Single Tree Accuracy: {single_accuracy:.3f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Improvement: {(accuracy_score(y_test, y_pred) - single_accuracy)*100:.1f}%")
```

## Best Practices

### Model Training
1. **Start with defaults**: Scikit-learn defaults work well
2. **Use OOB scoring**: Efficient hyperparameter tuning
3. **Tune n_estimators**: More trees generally better (diminishing returns)
4. **Set max_depth**: Prevent overfitting
5. **Use max_features='sqrt'**: Good balance for most datasets

### Feature Engineering
1. **Handle categorical features**: One-hot encoding or label encoding
2. **Scale features**: Not required but can help with other preprocessing
3. **Feature selection**: Use importance scores to remove irrelevant features
4. **Missing values**: Random Forest can handle some missing values

### Production Considerations
1. **Model persistence**: Save trained model with joblib
2. **Prediction speed**: Consider fewer trees for real-time applications
3. **Memory usage**: Monitor memory consumption with large forests
4. **Scalability**: Use distributed training for very large datasets

## Next Steps

This lecture covers Random Forest, one of the most powerful ensemble methods. The next lecture (14.10) will explore RAPIDS acceleration specifically for Random Forest, showing how GPU computing can dramatically speed up training and prediction.