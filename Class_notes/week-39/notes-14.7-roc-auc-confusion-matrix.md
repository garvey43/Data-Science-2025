# Lecture 14.7: ROC-AUC and Confusion Matrix

## Key Learning Objectives
- Understand confusion matrix components and interpretation
- Master ROC curves and AUC calculation
- Learn when to use different evaluation metrics
- Implement comprehensive model evaluation strategies

## Core Concepts

### Confusion Matrix

#### Basic Structure
```
Predicted →    Negative    Positive
Actual ↓
Negative        TN          FP
Positive        FN          TP
```

- **True Positive (TP)**: Correctly predicted positive cases
- **True Negative (TN)**: Correctly predicted negative cases
- **False Positive (FP)**: Incorrectly predicted positive (Type I error)
- **False Negative (FN)**: Incorrectly predicted negative (Type II error)

#### Implementation
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

## Classification Metrics

### Basic Metrics

#### Accuracy
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
# Overall correctness of predictions
# Not suitable for imbalanced datasets
```

#### Precision (Positive Predictive Value)
```python
precision = TP / (TP + FP)
# Of all positive predictions, how many were correct?
# Important when FP cost is high (e.g., spam detection)
```

#### Recall (Sensitivity, True Positive Rate)
```python
recall = TP / (TP + FN)
# Of all actual positives, how many were identified?
# Important when FN cost is high (e.g., disease detection)
```

#### Specificity (True Negative Rate)
```python
specificity = TN / (TN + FP)
# Of all actual negatives, how many were identified?
# Important for balanced evaluation
```

#### F1-Score
```python
f1 = 2 * (precision * recall) / (precision + recall)
# Harmonic mean of precision and recall
# Good for imbalanced datasets
```

### Implementation with scikit-learn
```python
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Individual metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Comprehensive report
print(classification_report(y_true, y_pred))
```

## ROC Curve and AUC

### Understanding ROC

#### ROC Curve Basics
- **ROC**: Receiver Operating Characteristic
- **X-axis**: False Positive Rate (FPR) = FP / (FP + TN) = 1 - Specificity
- **Y-axis**: True Positive Rate (TPR) = TP / (TP + FN) = Recall
- **Plot**: TPR vs FPR at different classification thresholds

#### Threshold Impact
- **High threshold**: Fewer positives predicted → Lower FPR, Lower TPR
- **Low threshold**: More positives predicted → Higher FPR, Higher TPR
- **Perfect classifier**: (0, 1) point on ROC curve
- **Random classifier**: Diagonal line from (0,0) to (1,1)

### AUC (Area Under Curve)
```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Get prediction probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# Alternative: Direct AUC calculation
auc_score = roc_auc_score(y_true, y_prob)

print(f"AUC Score: {auc_score:.3f}")
```

### ROC Curve Visualization
```python
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
```

## Threshold Selection

### Choosing the Right Threshold
```python
# Calculate metrics at different thresholds
thresholds = np.arange(0.1, 0.9, 0.1)
metrics = []

for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    precision = precision_score(y_true, y_pred_thresh)
    recall = recall_score(y_true, y_pred_thresh)
    f1 = f1_score(y_true, y_pred_thresh)
    
    metrics.append({
        'threshold': thresh,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Display results
metrics_df = pd.DataFrame(metrics)
print(metrics_df)
```

### Cost-Sensitive Threshold Selection
```python
# Define costs
cost_fp = 1  # Cost of false positive
cost_fn = 10  # Cost of false negative (more expensive)

# Calculate expected cost at each threshold
costs = []
for thresh in thresholds:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel()
    
    total_cost = fp * cost_fp + fn * cost_fn
    costs.append(total_cost)

# Find optimal threshold
optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold}")
```

## Multi-Class Evaluation

### One-vs-Rest ROC-AUC
```python
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Binarize labels for multi-class
y_bin = label_binarize(y_true, classes=np.unique(y_true))
n_classes = y_bin.shape[1]

# Calculate AUC for each class
auc_scores = []
for i in range(n_classes):
    auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
    auc_scores.append(auc_i)
    print(f"Class {i} AUC: {auc_i:.3f}")

# Macro-averaged AUC
macro_auc = np.mean(auc_scores)
print(f"Macro-averaged AUC: {macro_auc:.3f}")

# Overall AUC (one-vs-rest)
overall_auc = roc_auc_score(y_bin, y_prob, multi_class='ovr')
print(f"Overall AUC (OvR): {overall_auc:.3f}")
```

### Multi-Class Confusion Matrix
```python
from sklearn.metrics import multilabel_confusion_matrix

# Multi-class confusion matrix
mcm = multilabel_confusion_matrix(y_true, y_pred)

# Display each class's confusion matrix
for i, cm in enumerate(mcm):
    print(f"Class {i} Confusion Matrix:")
    print(cm)
    print()
```

## Model Comparison

### Comparing Multiple Models
```python
models = {
    'Logistic Regression': lr_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'SVM': svm_model
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('AUC', ascending=False))
```

### ROC Curve Comparison
```python
plt.figure(figsize=(10, 8))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
```

## Practical Considerations

### Imbalanced Datasets
```python
# For imbalanced datasets, focus on:
# - AUC (threshold-independent)
# - Precision-Recall curves
# - F1-score instead of accuracy

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_true, y_prob)
avg_precision = average_precision_score(y_true, y_prob)

plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### Cross-Validation with Proper Metrics
```python
from sklearn.model_selection import cross_validate

# Use multiple metrics in cross-validation
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

# Display results
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize()}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## Best Practices

### Metric Selection Guidelines
- **Balanced dataset**: Use accuracy
- **Imbalanced dataset**: Use F1-score, AUC
- **High FP cost**: Use precision
- **High FN cost**: Use recall
- **General purpose**: Use F1-score or AUC

### Evaluation Checklist
1. **Use appropriate metrics**: Consider business context
2. **Cross-validate**: Get robust performance estimates
3. **Check calibration**: Are probabilities meaningful?
4. **Consider threshold**: Default 0.5 may not be optimal
5. **Compare models**: Use same test set for fair comparison

### Common Pitfalls
- **Accuracy obsession**: Not suitable for imbalanced data
- **Ignoring costs**: Different errors have different costs
- **Single metric focus**: Use multiple complementary metrics
- **Test set leakage**: Never use test set for model selection

## Connection to Scalable Computing

### Distributed Evaluation
```python
# With Dask for large datasets
import dask_ml.metrics as dm

# Distributed confusion matrix
cm_distributed = dm.confusion_matrix(y_true_dask, y_pred_dask)

# Distributed AUC
auc_distributed = dm.roc_auc_score(y_true_dask, y_prob_dask)
```

### GPU-Accelerated Metrics
```python
# With RAPIDS cuML
from cuml.metrics import accuracy_score, confusion_matrix

# GPU confusion matrix
cm_gpu = confusion_matrix(y_true_gpu, y_pred_gpu)

# GPU AUC calculation
from cuml.metrics import roc_auc_score
auc_gpu = roc_auc_score(y_true_gpu, y_prob_gpu)
```

## Next Steps

This lecture covers comprehensive model evaluation techniques. The next lecture (14.8) will introduce bagging, the foundation of ensemble methods like Random Forests.