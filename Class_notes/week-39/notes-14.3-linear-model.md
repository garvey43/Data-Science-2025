# Lecture 14.3: Linear Models for Classification

## Key Learning Objectives
- Understand logistic regression and its mathematical foundation
- Learn about maximum likelihood estimation
- Master multi-class classification with softmax regression
- Implement linear classifiers with regularization

## Core Concepts

### From Linear Regression to Classification

#### Linear Regression Review
- **Equation**: ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
- **Problem**: Outputs continuous values, not suitable for classification
- **Solution**: Transform output to probability using sigmoid function

### Logistic Regression

#### The Sigmoid Function
- **Formula**: σ(z) = 1 / (1 + e^(-z))
- **Range**: [0, 1] - perfect for probability estimation
- **Interpretation**: Probability of positive class (y=1)

#### Logistic Regression Model
- **Raw output**: z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
- **Probability**: P(y=1|x) = σ(z) = 1 / (1 + e^(-z))
- **Decision Rule**: Classify as 1 if P(y=1|x) > 0.5, else 0

### Maximum Likelihood Estimation

#### Likelihood Function
- **Individual likelihood**: P(y|x; w) = σ(z)^y * (1-σ(z))^(1-y)
- **Log-likelihood**: log P(y|x; w) = y*log(σ(z)) + (1-y)*log(1-σ(z))
- **Cost function**: J(w) = -∑[y*log(σ(z)) + (1-y)*log(1-σ(z))]

#### Gradient Descent Optimization
- **Gradient**: ∂J/∂w = ∑(σ(z) - y)x
- **Update rule**: w := w - α * ∂J/∂w
- **Convergence**: Iterate until convergence or max iterations

## Multi-Class Classification

### One-vs-Rest (OvR) Approach
- **Strategy**: Train binary classifier for each class vs all others
- **Example**: For 3 classes (A, B, C):
  - Classifier 1: A vs (B,C)
  - Classifier 2: B vs (A,C)
  - Classifier 3: C vs (A,B)
- **Prediction**: Choose class with highest probability

### Softmax Regression (Multinomial Logistic Regression)
- **Generalization**: Extends logistic regression to multiple classes
- **Softmax function**: P(y=k|x) = e^(z_k) / ∑_{j=1}^K e^(z_j)
- **Properties**: All probabilities sum to 1, each in [0,1]

## Regularization Techniques

### L2 Regularization (Ridge)
- **Modified cost**: J(w) = -∑[y*log(σ(z)) + (1-y)*log(1-σ(z))] + λ∑w²
- **Effect**: Shrinks coefficients towards zero
- **Benefit**: Prevents overfitting, handles multicollinearity

### L1 Regularization (Lasso)
- **Modified cost**: J(w) = -∑[y*log(σ(z)) + (1-y)*log(1-σ(z))] + λ∑|w|
- **Effect**: Forces some coefficients to exactly zero
- **Benefit**: Feature selection, sparse models

### Elastic Net
- **Combination**: λ₁∑|w| + λ₂∑w²
- **Advantage**: Benefits of both L1 and L2 regularization

## Implementation in Python

### Basic Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create and train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

### Multi-Class Classification
```python
# Multi-class logistic regression (default)
multi_model = LogisticRegression(multi_class='ovr', random_state=42)
multi_model.fit(X_train, y_train)

# Or use softmax (multinomial)
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
softmax_model.fit(X_train, y_train)
```

### With Regularization
```python
# L2 regularization (default)
ridge_model = LogisticRegression(C=1.0, penalty='l2', random_state=42)

# L1 regularization
lasso_model = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', random_state=42)

# Elastic net (using SGDClassifier)
from sklearn.linear_model import SGDClassifier
elastic_model = SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.5, random_state=42)
```

## Model Interpretation

### Feature Importance
- **Coefficients**: Magnitude indicates feature importance
- **Sign**: Positive/negative relationship with target
- **Odds Ratio**: e^w represents change in odds for one unit increase

### Decision Boundaries
- **Linear boundary**: w₀ + w₁x₁ + w₂x₂ = 0
- **Visualization**: Plot decision boundary on feature space
- **Non-linear extensions**: Polynomial features, kernel methods

## Advantages and Limitations

### Advantages
- **Interpretable**: Clear relationship between features and outcome
- **Efficient**: Fast training and prediction
- **Probabilistic**: Provides probability estimates
- **Regularization**: Built-in overfitting prevention

### Limitations
- **Linearity assumption**: May not capture complex relationships
- **Sensitive to outliers**: Can be affected by extreme values
- **Multicollinearity**: Correlated features can cause instability
- **Class imbalance**: May favor majority class

## Performance Optimization

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

### Feature Scaling
- **Importance**: Logistic regression is sensitive to feature scales
- **Standardization**: Zero mean, unit variance
- **Min-Max scaling**: Scale to [0,1] range

## Connection to Scalable Computing

### Distributed Training
- **Spark MLlib**: Distributed logistic regression
- **Dask ML**: Parallel training on large datasets
- **RAPIDS**: GPU-accelerated training

### Large-Scale Applications
- **Online learning**: Stochastic gradient descent for streaming data
- **Distributed optimization**: Coordinate descent for large feature spaces
- **Model serving**: Deploy at scale with distributed systems

## Practical Considerations

### Data Preprocessing
1. **Handle missing values**: Imputation or removal
2. **Encode categorical variables**: One-hot encoding, label encoding
3. **Scale features**: Standardization for numerical features
4. **Handle class imbalance**: SMOTE, weighted classes

### Model Validation
1. **Cross-validation**: K-fold CV for robust evaluation
2. **Learning curves**: Check for overfitting/underfitting
3. **Feature selection**: Remove irrelevant features
4. **Multicollinearity check**: VIF, correlation analysis

## Common Pitfalls

### Don't Forget
- **Feature scaling**: Essential for convergence
- **Class balance**: Check for imbalanced datasets
- **Multicollinearity**: Can make coefficients unstable
- **Outliers**: Can significantly affect linear models

### Best Practices
- **Start simple**: Begin with default parameters
- **Regularization**: Use L2 by default, L1 for feature selection
- **Cross-validation**: Always use for hyperparameter tuning
- **Interpretability**: Consider coefficient magnitudes and signs

## Next Steps

This lecture covers linear models for classification. The next lecture (14.4) will explore RAPIDS acceleration for linear models, showing how GPU computing can dramatically speed up training and prediction.