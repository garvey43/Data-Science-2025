# Lecture 14.2: Introduction to Supervised Learning

## Key Learning Objectives
- Understand the fundamentals of supervised learning
- Learn about training, validation, and test sets
- Recognize overfitting and underfitting patterns
- Master data splitting strategies for model development

## Core Concepts

### What is Supervised Learning?
- **Definition**: Learning from labeled training data to make predictions
- **Key Components**:
  - **Features (X)**: Input variables used for prediction
  - **Target/Labels (y)**: Output variables we want to predict
  - **Model**: Mathematical function that maps X to y

### The Learning Process
1. **Training Phase**: Model learns patterns from labeled data
2. **Validation Phase**: Model is tuned and optimized
3. **Testing Phase**: Model performance is evaluated on unseen data

## Data Splitting Strategies

### Basic Train/Test Split
```python
from sklearn.model_selection import train_test_split

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Three-Way Split (Train/Validation/Test)
```python
# 60% training, 20% validation, 20% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

### Cross-Validation
- **K-Fold CV**: Data divided into k subsets, each used as test set once
- **Stratified K-Fold**: Maintains class distribution in each fold
- **Leave-One-Out**: Each sample used as test set once

## Overfitting vs Underfitting

### Overfitting
- **Definition**: Model performs well on training data but poorly on unseen data
- **Symptoms**:
  - High training accuracy, low test accuracy
  - Complex models that memorize training data
  - Poor generalization to new data

### Underfitting
- **Definition**: Model performs poorly on both training and test data
- **Symptoms**:
  - Low accuracy on both training and test sets
  - Model too simple to capture underlying patterns
  - High bias, low variance

### Bias-Variance Tradeoff
- **Bias**: Error from incorrect assumptions (underfitting)
- **Variance**: Error from sensitivity to training data variations (overfitting)
- **Goal**: Find optimal balance between bias and variance

## Model Complexity and Performance

### Learning Curves
- **Training Learning Curve**: Performance on training set vs training size
- **Validation Learning Curve**: Performance on validation set vs training size
- **Analysis**:
  - High gap between curves indicates overfitting
  - Both curves plateauing low indicates underfitting
  - Converging curves indicate good generalization

### Regularization Techniques
- **L1 Regularization (Lasso)**: Feature selection, sparse models
- **L2 Regularization (Ridge)**: Prevents overfitting, shrinks coefficients
- **Elastic Net**: Combination of L1 and L2 regularization

## Evaluation Metrics for Classification

### Basic Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP) - Positive predictive value
- **Recall**: TP / (TP + FN) - True positive rate
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)

### Confusion Matrix Components
- **True Positive (TP)**: Correctly predicted positive
- **True Negative (TN)**: Correctly predicted negative
- **False Positive (FP)**: Incorrectly predicted positive (Type I error)
- **False Negative (FN)**: Incorrectly predicted negative (Type II error)

## Practical Implementation

### Data Preparation
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
df = pd.read_csv('dataset.csv')

# Handle missing values
df = df.dropna()

# Encode categorical variables
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['feature1', 'feature2', 'feature3']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

### Model Training Pipeline
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## Common Challenges

### Data Imbalance
- **Problem**: Unequal class distribution
- **Solutions**:
  - Oversampling minority class
  - Undersampling majority class
  - Using weighted loss functions
  - SMOTE (Synthetic Minority Oversampling Technique)

### Feature Engineering
- **Importance**: Quality of features affects model performance
- **Techniques**:
  - Feature scaling/normalization
  - Categorical encoding
  - Feature interaction terms
  - Dimensionality reduction

## Connection to Scalable Computing

### Distributed Learning Considerations
- **Large Datasets**: Use Spark/Dask for data preprocessing
- **Model Training**: Parallel training with distributed frameworks
- **Cross-Validation**: Implement distributed CV for large data
- **GPU Acceleration**: RAPIDS for faster training on large datasets

## Best Practices

### Data Splitting
1. **Random Split**: For independent, identically distributed data
2. **Stratified Split**: Maintain class proportions
3. **Time-Based Split**: For time-series data
4. **Group-Based Split**: Prevent data leakage in grouped data

### Model Validation
1. **Always use separate test set**: Never evaluate on training data
2. **Use cross-validation**: Get robust performance estimates
3. **Monitor for overfitting**: Compare train vs validation performance
4. **Use appropriate metrics**: Choose metrics relevant to business problem

## Next Steps

This lecture establishes the foundation for understanding supervised learning. The next lecture (14.3) will dive into linear models for classification, including logistic regression and its variants.