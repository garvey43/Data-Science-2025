# Lecture 14.5: Overfitting and Cross-Validation

## Key Learning Objectives
- Understand overfitting and underfitting in classification models
- Master cross-validation techniques for robust model evaluation
- Learn regularization methods to prevent overfitting
- Implement proper model validation strategies

## Core Concepts

### Overfitting vs Underfitting

#### Overfitting
- **Definition**: Model learns noise and specific patterns in training data that don't generalize
- **Symptoms**:
  - High training accuracy, low test accuracy
  - Complex decision boundaries that follow training points exactly
  - Poor performance on unseen data

#### Underfitting
- **Definition**: Model is too simple to capture underlying patterns
- **Symptoms**:
  - Low accuracy on both training and test sets
  - Model doesn't fit training data well
  - High bias, low variance

#### Bias-Variance Tradeoff
- **Bias**: Error from incorrect assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- **Irreducible Error**: Noise inherent in the data

### Visualizing Model Complexity

#### Learning Curves
```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    estimator, X, y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.legend()
plt.show()
```

#### Interpretation
- **Converging curves**: Good generalization
- **Large gap**: Overfitting
- **Both low**: Underfitting

## Cross-Validation Techniques

### K-Fold Cross-Validation

#### Basic K-Fold CV
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

#### Stratified K-Fold
- **Purpose**: Maintains class distribution in each fold
- **Important for**: Imbalanced datasets
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
```

### Other Cross-Validation Methods

#### Leave-One-Out Cross-Validation (LOOCV)
```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
# Very expensive for large datasets
```

#### Time Series Split
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
```

## Regularization Techniques

### L2 Regularization (Ridge)
- **Cost function**: J(w) = Loss + λ∑w²
- **Effect**: Shrinks coefficients towards zero
- **sklearn**: `penalty='l2'`, `C=1/λ`

### L1 Regularization (Lasso)
- **Cost function**: J(w) = Loss + λ∑|w|
- **Effect**: Forces some coefficients to exactly zero (feature selection)
- **sklearn**: `penalty='l1'`, `solver='liblinear'`

### Elastic Net
- **Cost function**: J(w) = Loss + λ₁∑|w| + λ₂∑w²
- **Benefits**: Combines L1 and L2 properties
- **sklearn**: `penalty='elasticnet'`, `l1_ratio`

## Hyperparameter Tuning

### Grid Search Cross-Validation
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

### Randomized Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

param_dist = {
    'C': loguniform(1e-4, 1e2),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

random_search = RandomizedSearchCV(
    LogisticRegression(random_state=42),
    param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

## Model Evaluation Metrics

### Classification Metrics

#### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

#### Key Metrics
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Specificity**: TN / (TN + FP)

### ROC-AUC for Binary Classification
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

## Practical Implementation

### Complete Model Validation Pipeline
```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and prepare data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with cross-validation
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Final evaluation on test set
y_pred = best_model.predict(X_test_scaled)
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Handling Class Imbalance

### Techniques for Imbalanced Data
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# SMOTE oversampling
smote = SMOTE(random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', smote),
    ('classifier', LogisticRegression(random_state=42))
])

# Cross-validation with SMOTE
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
print(f"CV F1 scores with SMOTE: {scores}")
print(f"Mean F1: {scores.mean():.3f}")
```

### Class Weights
```python
# Automatic class weighting
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Manual class weights
class_weights = {0: 1, 1: 5}  # Give class 1 higher weight
model = LogisticRegression(class_weight=class_weights, random_state=42)
```

## Scalable Cross-Validation

### With Dask for Large Datasets
```python
import dask_ml.model_selection as dcv
from dask_ml.linear_model import LogisticRegression as DaskLogisticRegression

# Distributed cross-validation
dask_model = DaskLogisticRegression(random_state=42)
scores = dcv.cross_val_score(dask_model, X_dask, y_dask, cv=5)
print(f"Dask CV scores: {scores.compute()}")
```

### With RAPIDS for GPU Acceleration
```python
from cuml.model_selection import train_test_split as cuml_split
from cuml.linear_model import LogisticRegression as CumlLogisticRegression

# GPU-accelerated cross-validation
X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu = cuml_split(X_gpu, y_gpu, test_size=0.2)

gpu_model = CumlLogisticRegression()
gpu_model.fit(X_train_gpu, y_train_gpu)

# Evaluate on GPU
from cuml.metrics import accuracy_score
y_pred_gpu = gpu_model.predict(X_test_gpu)
accuracy = accuracy_score(y_test_gpu, y_pred_gpu)
```

## Best Practices

### Model Validation Checklist
1. **Split data properly**: Use stratified split for imbalanced data
2. **Use cross-validation**: Don't rely on single train/test split
3. **Choose appropriate metrics**: Consider business context
4. **Tune hyperparameters**: Use grid/random search with CV
5. **Validate on test set**: Final evaluation on unseen data
6. **Check for overfitting**: Compare train vs validation performance

### Common Pitfalls to Avoid
- **Data leakage**: Ensure no test information in training
- **Over-optimistic CV**: Using test set for model selection
- **Ignoring class imbalance**: Check class distribution
- **Single metric focus**: Use multiple evaluation metrics
- **Insufficient CV folds**: Use at least 5-fold CV

## Next Steps

This lecture covers model validation and overfitting prevention. The next lecture (14.6) will introduce decision trees, a powerful non-linear classification algorithm that forms the foundation for ensemble methods.