# Lecture 14.11: Boosting Methods for Classification

## Key Learning Objectives
- Understand the boosting ensemble paradigm
- Learn AdaBoost and Gradient Boosting algorithms
- Master XGBoost implementation and optimization
- Implement boosting with regularization and early stopping

## Core Concepts

### What is Boosting?

#### Sequential Ensemble Learning
- **Sequential training**: Models built one after another
- **Error correction**: Each model focuses on previous errors
- **Weighted combination**: Models combined with learned weights
- **Adaptive learning**: Algorithm adapts to difficult examples

#### Boosting vs Bagging
| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Focus | Variance reduction | Bias reduction |
| Sample weighting | Equal weights | Adaptive weights |
| Error handling | Independent errors | Correct previous errors |
| Example | Random Forest | XGBoost |

### AdaBoost (Adaptive Boosting)

#### Algorithm Overview
1. **Initialize weights**: Equal weights for all training samples
2. **Train weak learner**: Fit a simple model (e.g., decision stump)
3. **Calculate error**: Compute weighted error of the model
4. **Compute model weight**: Higher weight for better models
5. **Update sample weights**: Increase weights for misclassified samples
6. **Repeat**: Train next model on updated weights
7. **Final prediction**: Weighted majority vote

#### Mathematical Foundation
```python
# AdaBoost algorithm pseudocode
def adaboost(X, y, T):
    # Initialize sample weights
    w = np.ones(len(X)) / len(X)
    
    models = []
    alphas = []
    
    for t in range(T):
        # Train weak learner with current weights
        model = train_weak_learner(X, y, w)
        
        # Calculate weighted error
        predictions = model.predict(X)
        error = np.sum(w * (predictions != y)) / np.sum(w)
        
        # Calculate model weight
        alpha = 0.5 * np.log((1 - error) / error)
        
        # Update sample weights
        w = w * np.exp(-alpha * y * predictions)
        w = w / np.sum(w)  # Normalize
        
        models.append(model)
        alphas.append(alpha)
    
    return models, alphas

# Final prediction
def predict_adaboost(models, alphas, X):
    predictions = np.zeros(len(X))
    for model, alpha in zip(models, alphas):
        predictions += alpha * model.predict(X)
    return np.sign(predictions)
```

## Gradient Boosting

### Gradient Descent Perspective

#### Loss Function Optimization
- **Loss function**: Measures prediction error (e.g., log loss for classification)
- **Gradient**: Direction of steepest ascent in loss function
- **Pseudo-residuals**: Negative gradient becomes target for next model
- **Additive model**: F(x) = F₀(x) + α₁h₁(x) + α₂h₂(x) + ...

#### Algorithm Steps
1. **Initialize**: Start with constant prediction (e.g., log(odds) for classification)
2. **Compute pseudo-residuals**: rᵢ = -∂L/∂F(xᵢ) where L is loss function
3. **Fit base learner**: Train model h on pseudo-residuals
4. **Compute multiplier**: α = argmin_α ∑L(yᵢ, F(xᵢ) + αh(xᵢ))
5. **Update model**: F := F + αh
6. **Repeat**: Until convergence or max iterations

### XGBoost (Extreme Gradient Boosting)

#### Advanced Features
- **Regularization**: L1 and L2 regularization terms
- **Tree pruning**: Post-pruning with complexity control
- **Sparsity handling**: Automatic handling of missing values
- **Parallel processing**: Parallel tree construction
- **Cache optimization**: Efficient memory usage

#### XGBoost Objective
```
Obj = ∑[l(yᵢ, ŷᵢ)] + ∑[Ω(fₜ)]  (for t=1 to T trees)

Where:
- l(yᵢ, ŷᵢ): Loss function (log loss for classification)
- Ω(f): Regularization term = γT + (1/2)λ∑w²
- γ: Minimum loss reduction required for split
- λ: L2 regularization on leaf weights
- T: Number of leaves
```

## Implementation

### AdaBoost with Scikit-Learn
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create AdaBoost classifier
ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Decision stump
    n_estimators=50,        # Number of weak learners
    learning_rate=1.0,      # Contribution of each model
    algorithm='SAMME.R',    # Real-valued predictions
    random_state=42
)

# Train the model
ada.fit(X_train, y_train)

# Make predictions
y_pred = ada.predict(X_test)
y_prob = ada.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

# Feature importance (from base estimators)
feature_importance = np.mean([estimator.feature_importances_ for estimator in ada.estimators_], axis=0)
```

### XGBoost Implementation
```python
import xgboost as xgb
from xgboost import XGBClassifier

# Create XGBoost classifier
xgb_model = XGBClassifier(
    n_estimators=100,       # Number of boosting rounds
    max_depth=6,            # Maximum tree depth
    learning_rate=0.3,      # Step size shrinkage
    subsample=0.8,          # Subsample ratio for training
    colsample_bytree=0.8,   # Subsample ratio of columns
    gamma=0,                # Minimum loss reduction for split
    reg_alpha=0,            # L1 regularization
    reg_lambda=1,           # L2 regularization
    random_state=42,
    n_jobs=-1               # Use all cores
)

# Train with early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='logloss',
    early_stopping_rounds=10,
    verbose=True
)

# Make predictions
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Best iteration: {xgb_model.best_iteration}")
```

### Gradient Boosting with Scikit-Learn
```python
from sklearn.ensemble import GradientBoostingClassifier

# Create Gradient Boosting classifier
gb = GradientBoostingClassifier(
    n_estimators=100,       # Number of boosting stages
    max_depth=3,            # Maximum tree depth
    learning_rate=0.1,      # Learning rate
    subsample=1.0,          # Subsample ratio
    criterion='friedman_mse',  # Splitting criterion
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Train the model
gb.fit(X_train, y_train)

# Feature importance
plt.figure(figsize=(10, 6))
feature_importance = gb.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting Feature Importance')
plt.show()
```

## Hyperparameter Tuning

### XGBoost Parameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Train final model with best parameters
best_xgb = grid_search.best_estimator_
```

### Learning Rate and Early Stopping
```python
# Use learning rate decay with early stopping
xgb_model = XGBClassifier(
    n_estimators=1000,      # Large number, early stopping will limit
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# Train with early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='logloss',
    early_stopping_rounds=50,   # Stop if no improvement for 50 rounds
    verbose=50
)

print(f"Best iteration: {xgb_model.best_iteration}")
print(f"Best score: {xgb_model.best_score:.3f}")
```

## Handling Class Imbalance

### Scale Positive Weight
```python
# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_balanced = XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # Handle class imbalance
    random_state=42
)

xgb_balanced.fit(X_train, y_train)
```

### Sample Weight
```python
# Use sample weights
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

xgb_weighted = XGBClassifier(random_state=42)
xgb_weighted.fit(X_train, y_train, sample_weight=sample_weights)
```

## Model Interpretation

### Feature Importance Types
```python
# Different importance types in XGBoost
importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

for imp_type in importance_types:
    importance = xgb_model.get_booster().get_score(importance_type=imp_type)
    print(f"\n{imp_type.upper()} importance:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {score:.3f}")
```

### SHAP Values for Interpretability
```python
import shap

# Calculate SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# Waterfall plot for single prediction
shap.plots.waterfall(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

## Performance Optimization

### XGBoost Best Practices
1. **Start with defaults**: XGBoost defaults work well
2. **Tune learning rate first**: Lower learning rate with more trees
3. **Use early stopping**: Prevent overfitting
4. **Tune tree-specific parameters**: max_depth, min_child_weight
5. **Tune regularization**: lambda, alpha, gamma

### Memory Optimization
```python
# For large datasets
dtrain = xgb.DMatrix(X_train, label=y_train)

params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',  # Faster histogram-based method
    'grow_policy': 'lossguide'  # Grow trees by loss
}

# Train with early stopping
bst = xgb.train(
    params, 
    dtrain, 
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (xgb.DMatrix(X_test, label=y_test), 'test')],
    early_stopping_rounds=50,
    verbose_eval=50
)
```

## Comparison of Boosting Methods

### Performance Comparison
```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
import time

models = {
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=50, random_state=42)
}

results = []
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Training Time': train_time
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('Accuracy', ascending=False))
```

## Practical Implementation

### Complete Boosting Pipeline
```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
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

# XGBoost with comprehensive setup
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

# Train with evaluation monitoring
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    eval_metric=['logloss', 'auc'],
    early_stopping_rounds=20,
    verbose=True
)

# Model evaluation
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

print("=== Model Performance ===")
print(f"Best iteration: {xgb_model.best_iteration}")
print(f"Training stopped at: {xgb_model.best_ntree_limit} trees")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Test AUC: {roc_auc_score(y_test, y_prob):.3f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Feature importance
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title('XGBoost Feature Importance')
plt.show()

# Learning curves
results = xgb_model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_axis, results['validation_0']['auc'], label='Train')
plt.plot(x_axis, results['validation_1']['auc'], label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('AUC')
plt.title('XGBoost AUC')
plt.legend()

plt.tight_layout()
plt.show()

# Cross-validation for robust evaluation
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Model comparison with other boosting methods
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

models = {
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=50, random_state=42)
}

comparison_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_comp = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_comp)
    
    comparison_results.append({
        'Model': name,
        'Accuracy': accuracy
    })

comparison_df = pd.DataFrame(comparison_results)
print("\n=== Model Comparison ===")
print(comparison_df.sort_values('Accuracy', ascending=False))

# Save model
xgb_model.save_model('xgboost_model.json')
print("\nModel saved as 'xgboost_model.json'")
```

## Best Practices

### Boosting Guidelines
1. **Start simple**: Use default parameters first
2. **Tune learning rate**: Smaller learning rate with more trees
3. **Use early stopping**: Prevent overfitting
4. **Monitor validation**: Track performance on unseen data
5. **Regularize**: Use appropriate regularization parameters

### Common Pitfalls
- **Overfitting**: Too many trees without early stopping
- **Slow training**: Large learning rate with few trees
- **Poor generalization**: No validation set monitoring
- **Memory issues**: Large max_depth with many features
- **Class imbalance**: Not handling imbalanced datasets

## Next Steps

This lecture covers boosting methods for classification. The next lecture (14.12) will explore XGBoost with RAPIDS acceleration, combining the power of boosting with GPU computing for maximum performance.