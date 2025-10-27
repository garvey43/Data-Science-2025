# Lecture 14.8: Bagging (Bootstrap Aggregation)

## Key Learning Objectives
- Understand the bagging ensemble method and its benefits
- Learn how bootstrap sampling reduces variance
- Master out-of-bag (OOB) error estimation
- Implement bagging with scikit-learn and RAPIDS

## Core Concepts

### What is Bagging?

#### Bootstrap Aggregation
- **Bootstrap**: Random sampling with replacement
- **Aggregation**: Combine predictions from multiple models
- **Goal**: Reduce variance while maintaining low bias

#### Why Bagging Works
- **Variance Reduction**: Averaging reduces individual model variance
- **Stability**: Less sensitive to training data variations
- **Overfitting Prevention**: Each model sees different data subset

### Bootstrap Sampling

#### Basic Concept
```python
import numpy as np

def bootstrap_sample(X, y, n_samples=None):
    """
    Create a bootstrap sample from the dataset
    """
    if n_samples is None:
        n_samples = len(X)
    
    indices = np.random.choice(len(X), size=n_samples, replace=True)
    return X[indices], y[indices]

# Example usage
X_boot, y_boot = bootstrap_sample(X_train, y_train)
print(f"Original size: {len(X_train)}")
print(f"Bootstrap size: {len(X_boot)}")
print(f"Unique samples: {len(np.unique(y_boot))}")  # May be less than original
```

#### Bootstrap Properties
- **Same size**: Bootstrap sample has same size as original
- **With replacement**: Same sample can appear multiple times
- **Approximately 63.2% unique**: On average, ~63.2% of original samples appear in bootstrap

## Bagging Algorithm

### Basic Bagging Implementation
```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class BaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator or DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Create bootstrap sample
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train base estimator
            estimator = self._clone_estimator()
            estimator.fit(X_boot, y_boot)
            self.estimators_.append(estimator)
        
        return self
    
    def predict(self, X):
        # Get predictions from all estimators
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        # Majority vote for classification
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    
    def _clone_estimator(self):
        from sklearn.base import clone
        return clone(self.base_estimator)
```

### Using Scikit-Learn Bagging
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Create bagging classifier
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,        # Number of base estimators
    max_samples=0.8,         # Fraction of samples for each bootstrap
    max_features=1.0,        # Fraction of features to use
    bootstrap=True,          # Use bootstrap sampling
    bootstrap_features=False, # Don't bootstrap features
    random_state=42
)

# Train the model
bagging.fit(X_train, y_train)

# Make predictions
y_pred = bagging.predict(X_test)
y_prob = bagging.predict_proba(X_test)
```

## Out-of-Bag (OOB) Error Estimation

### OOB Concept
- **Out-of-bag samples**: Samples not included in bootstrap (approximately 36.8%)
- **OOB prediction**: Use only trees that didn't see the sample during training
- **Advantage**: Built-in cross-validation without additional computation

### OOB Implementation
```python
# Enable OOB scoring
bagging_oob = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    oob_score=True,          # Enable OOB scoring
    random_state=42
)

bagging_oob.fit(X_train, y_train)

# Access OOB score
print(f"OOB Score: {bagging_oob.oob_score_:.3f}")

# OOB decision function (probabilities)
oob_prob = bagging_oob.oob_decision_function_
print(f"OOB probabilities shape: {oob_prob.shape}")
```

### Manual OOB Calculation
```python
def calculate_oob_predictions(bagging_model, X_train):
    """
    Manually calculate OOB predictions
    """
    n_samples = len(X_train)
    oob_predictions = np.zeros((n_samples, len(np.unique(y_train))))
    
    for i, estimator in enumerate(bagging_model.estimators_):
        # Find samples not used in training this estimator
        # (This requires tracking bootstrap indices during training)
        pass
    
    # This is simplified - actual implementation tracks bootstrap indices
    return oob_predictions
```

## Feature Importance in Bagging

### Aggregated Feature Importance
```python
# Get feature importance from each tree
feature_importances = np.array([tree.feature_importances_ for tree in bagging.estimators_])

# Average across all trees
avg_importance = np.mean(feature_importances, axis=0)
std_importance = np.std(feature_importances, axis=0)

# Display results
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': avg_importance,
    'importance_std': std_importance
}).sort_values('importance_mean', ascending=False)

print(importance_df.head(10))
```

## Bagging with RAPIDS

### GPU-Accelerated Bagging
```python
import cudf
from cuml.ensemble import RandomForestClassifier as CumlRandomForest

# Convert to GPU dataframes
X_gpu = cudf.DataFrame.from_pandas(X_train)
y_gpu = cudf.Series(y_train)

# RAPIDS Random Forest (ensemble of decision trees)
gpu_rf = CumlRandomForest(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

gpu_rf.fit(X_gpu, y_gpu)

# Predictions
y_pred_gpu = gpu_rf.predict(X_gpu)
y_prob_gpu = gpu_rf.predict_proba(X_gpu)
```

### Performance Comparison
```python
import time

# CPU Bagging
start_time = time.time()
cpu_bagging = BaggingClassifier(n_estimators=100, random_state=42)
cpu_bagging.fit(X_train, y_train)
cpu_time = time.time() - start_time

# GPU Random Forest
start_time = time.time()
gpu_rf.fit(X_gpu, y_gpu)
gpu_time = time.time() - start_time

print(f"CPU time: {cpu_time:.2f}s")
print(f"GPU time: {gpu_time:.2f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

## Advanced Bagging Techniques

### Pasting (Bootstrap without Replacement)
```python
# Pasting: sampling without replacement
pasting = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    bootstrap=False,  # No replacement
    random_state=42
)
```

### Random Subspaces
```python
# Random subspaces: feature bagging
subspace_bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_features=0.5,       # Use only 50% of features
    bootstrap_features=True, # Bootstrap features
    random_state=42
)
```

### Random Patches
```python
# Random patches: both sample and feature bagging
patches_bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,        # Sample bagging
    max_features=0.5,       # Feature bagging
    bootstrap_features=True,
    random_state=42
)
```

## Hyperparameter Tuning

### Key Parameters to Tune
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.5, 0.7, 1.0],
    'base_estimator__max_depth': [5, 10, None],
    'base_estimator__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42), random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

## Advantages and Limitations

### Advantages
- **Variance Reduction**: Significantly reduces overfitting
- **Parallel Training**: Base models can be trained independently
- **OOB Estimation**: Built-in cross-validation
- **Robustness**: Less sensitive to training data variations
- **Feature Importance**: Aggregated importance across models

### Limitations
- **Bias Preservation**: Doesn't reduce bias of base models
- **Computational Cost**: Training multiple models
- **Memory Usage**: Stores multiple models
- **Interpretability**: Harder to interpret than single models

## Practical Implementation

### Complete Bagging Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
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

# Single decision tree (baseline)
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_pred = single_tree.predict(X_test)
single_score = accuracy_score(y_test, single_pred)

# Bagging ensemble
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    oob_score=True,
    random_state=42
)

bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)
bagging_score = accuracy_score(y_test, bagging_pred)

print(f"Single Tree Accuracy: {single_score:.3f}")
print(f"Bagging Accuracy: {bagging_score:.3f}")
print(f"OOB Score: {bagging.oob_score_:.3f}")

# Cross-validation comparison
single_cv = cross_val_score(single_tree, X_train, y_train, cv=5)
bagging_cv = cross_val_score(bagging, X_train, y_train, cv=5)

print(f"Single Tree CV: {single_cv.mean():.3f} (+/- {single_cv.std()*2:.3f})")
print(f"Bagging CV: {bagging_cv.mean():.3f} (+/- {bagging_cv.std()*2:.3f})")

# Feature importance
feature_importance = np.mean([tree.feature_importances_ for tree in bagging.estimators_], axis=0)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 5 Important Features:")
print(importance_df.head())

# Plot learning curve
n_estimators_range = [10, 25, 50, 100, 200]
train_scores = []
val_scores = []

for n_est in n_estimators_range:
    bag = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n_est,
        random_state=42
    )
    
    # Simple train/validation split for demonstration
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    bag.fit(X_tr, y_tr)
    train_pred = bag.predict(X_tr)
    val_pred = bag.predict(X_val)
    
    train_scores.append(accuracy_score(y_tr, train_pred))
    val_scores.append(accuracy_score(y_val, val_pred))

plt.plot(n_estimators_range, train_scores, label='Training Accuracy')
plt.plot(n_estimators_range, val_scores, label='Validation Accuracy')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Bagging Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
```

## Best Practices

### When to Use Bagging
1. **High variance models**: Decision trees, neural networks
2. **Sufficient data**: Need enough data for bootstrap sampling
3. **Computational resources**: Can parallelize training
4. **Prediction stability**: Want robust predictions

### Optimization Tips
1. **Start with 100 estimators**: Good balance of performance vs computation
2. **Use OOB scoring**: Efficient hyperparameter tuning
3. **Tune base estimator**: Optimize the base model first
4. **Consider max_samples**: 0.8 often works well
5. **Monitor overfitting**: Check training vs validation performance

## Next Steps

This lecture covers bagging, the foundation of ensemble methods. The next lecture (14.9) will introduce Random Forest, which combines bagging with random feature selection for even better performance.