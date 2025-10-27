# Lecture 14.6: Decision Trees for Classification

## Key Learning Objectives
- Understand how decision trees work for classification
- Learn tree construction algorithms (ID3, C4.5, CART)
- Master tree pruning techniques to prevent overfitting
- Implement decision trees with scikit-learn and RAPIDS

## Core Concepts

### What is a Decision Tree?

#### Tree Structure
- **Root Node**: Starting point, contains all training data
- **Internal Nodes**: Decision points based on feature values
- **Leaf Nodes**: Final predictions (class labels)
- **Branches**: Represent decision outcomes

#### Decision Process
1. Start at root node
2. Evaluate feature condition
3. Follow appropriate branch
4. Repeat until reaching leaf node
5. Return class prediction

### Tree Construction Algorithms

#### ID3 (Iterative Dichotomiser 3)
- **Splitting Criterion**: Information Gain
- **Formula**: IG = Entropy(parent) - ∑(weighted_avg) * Entropy(children)
- **Limitation**: Biased towards features with many values

#### C4.5 (Successor to ID3)
- **Improvements**: Handles continuous features, missing values
- **Splitting Criterion**: Information Gain Ratio
- **Formula**: GR = IG / SplitInfo
- **SplitInfo**: Measures how evenly data is distributed

#### CART (Classification and Regression Trees)
- **Splitting Criterion**: Gini Impurity (classification)
- **Formula**: Gini = 1 - ∑(p_i²) for all classes i
- **Advantages**: Handles both classification and regression

## Splitting Criteria

### Gini Impurity
```python
def gini_impurity(y):
    """
    Calculate Gini impurity for a set of labels
    """
    if len(y) == 0:
        return 0
    
    p = np.bincount(y) / len(y)  # Class probabilities
    return 1 - np.sum(p ** 2)

# Example
y = [0, 0, 1, 1, 1]
gini = gini_impurity(y)
print(f"Gini impurity: {gini:.3f}")  # 0.48
```

### Information Gain (Entropy)
```python
def entropy(y):
    """
    Calculate entropy for a set of labels
    """
    if len(y) == 0:
        return 0
    
    p = np.bincount(y) / len(y)
    p = p[p > 0]  # Remove zero probabilities
    return -np.sum(p * np.log2(p))

def information_gain(y_parent, y_left, y_right):
    """
    Calculate information gain for a split
    """
    n = len(y_parent)
    n_left, n_right = len(y_left), len(y_right)
    
    ig = entropy(y_parent) - (n_left/n * entropy(y_left) + n_right/n * entropy(y_right))
    return ig
```

## Decision Tree Implementation

### Scikit-Learn Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create and train decision tree
dt = DecisionTreeClassifier(
    criterion='gini',      # 'gini' or 'entropy'
    max_depth=5,           # Maximum tree depth
    min_samples_split=2,   # Minimum samples to split
    min_samples_leaf=1,    # Minimum samples per leaf
    random_state=42
)

dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

### Tree Visualization
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(
    dt,
    feature_names=X.columns.tolist(),
    class_names=['Class 0', 'Class 1'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()
```

## Preventing Overfitting

### Pre-Pruning (During Construction)
```python
# Limit tree growth
dt_pruned = DecisionTreeClassifier(
    max_depth=3,                    # Limit depth
    min_samples_split=10,           # Require more samples to split
    min_samples_leaf=5,             # Require samples per leaf
    max_features='sqrt',            # Use subset of features
    random_state=42
)
```

### Post-Pruning (Cost Complexity Pruning)
```python
# Cost complexity pruning
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Train trees with different alpha values
trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
    tree.fit(X_train, y_train)
    trees.append(tree)

# Find optimal alpha using cross-validation
from sklearn.model_selection import cross_val_score
scores = [cross_val_score(tree, X_train, y_train, cv=5).mean() for tree in trees]
optimal_alpha = ccp_alphas[np.argmax(scores)]

# Train final pruned tree
dt_pruned = DecisionTreeClassifier(ccp_alpha=optimal_alpha, random_state=42)
dt_pruned.fit(X_train, y_train)
```

## Feature Importance

### Gini Importance
```python
# Feature importance from Gini impurity reduction
feature_importance = dt.feature_importances_
feature_names = X.columns

# Sort and display
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10))
```

### Permutation Importance
```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(dt, X_test, y_test, n_repeats=10, random_state=42)

# Display results
perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print(perm_df.head(10))
```

## Handling Different Data Types

### Categorical Features
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label encoding for ordinal features
le = LabelEncoder()
X['ordinal_feature'] = le.fit_transform(X['ordinal_feature'])

# One-hot encoding for nominal features
ohe = OneHotEncoder(sparse=False)
encoded_features = ohe.fit_transform(X[['nominal_feature']])
encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out())
X = pd.concat([X.drop('nominal_feature', axis=1), encoded_df], axis=1)
```

### Missing Values
```python
# Decision trees can handle missing values in some implementations
# For scikit-learn, we need to handle missing values first
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_imputed, y)
```

## Decision Trees with RAPIDS

### GPU-Accelerated Decision Trees
```python
import cudf
from cuml.ensemble import RandomForestClassifier as CumlRandomForest
from cuml.preprocessing import LabelEncoder as CumlLabelEncoder

# Convert to GPU dataframes
X_gpu = cudf.DataFrame.from_pandas(X)
y_gpu = cudf.Series(y)

# GPU decision tree (using RandomForest with n_estimators=1)
gpu_dt = CumlRandomForest(n_estimators=1, random_state=42)
gpu_dt.fit(X_gpu, y_gpu)

# Predictions
y_pred_gpu = gpu_dt.predict(X_gpu)
```

## Advantages and Limitations

### Advantages
- **Interpretable**: Easy to understand and visualize
- **Handles mixed data**: Works with numerical and categorical features
- **Non-parametric**: No assumptions about data distribution
- **Feature selection**: Built-in feature importance
- **Robust to outliers**: Less affected by extreme values

### Limitations
- **Overfitting**: Tendency to overfit without pruning
- **Instability**: Small data changes can result in very different trees
- **Bias towards features**: Favors features with more categories
- **Not suitable for linear relationships**: Can't capture linear patterns

## Ensemble Methods Foundation

### Why Ensemble Methods Work
- **Wisdom of crowds**: Combining multiple models reduces variance
- **Bias-variance tradeoff**: Trees have low bias but high variance
- **Bootstrap aggregation**: Reduces variance through averaging

### Preview of Random Forests
- **Bootstrap sampling**: Each tree trained on random subset of data
- **Feature randomness**: Each split considers random subset of features
- **Voting**: Majority vote for classification

## Practical Implementation

### Complete Decision Tree Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
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

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Grid search with cross-validation
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model
best_dt = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Final evaluation
y_pred = best_dt.predict(X_test)
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_dt.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head())

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(
    best_dt,
    feature_names=X.columns.tolist(),
    class_names=[str(cls) for cls in np.unique(y)],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Structure")
plt.show()
```

## Best Practices

### Tree Construction
1. **Start simple**: Begin with default parameters
2. **Use cross-validation**: For hyperparameter tuning
3. **Prune aggressively**: Prevent overfitting
4. **Consider ensemble methods**: For better performance

### Feature Engineering
1. **Handle categorical features**: Proper encoding
2. **Deal with missing values**: Appropriate imputation
3. **Feature scaling**: Not required for trees, but helps with other methods
4. **Feature selection**: Use importance scores

## Next Steps

This lecture covers decision trees, the building blocks of ensemble methods. The next lecture (14.7) will explore ROC-AUC curves and confusion matrices, essential tools for evaluating classification model performance.