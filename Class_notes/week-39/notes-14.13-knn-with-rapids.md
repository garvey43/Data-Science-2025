# Lecture 14.13: KNN with RAPIDS Acceleration

## Key Learning Objectives
- Understand k-Nearest Neighbors algorithm for classification
- Learn GPU acceleration with RAPIDS cuML KNN
- Compare CPU vs GPU KNN performance
- Implement scalable KNN for large datasets

## Core Concepts

### What is k-Nearest Neighbors (KNN)?

#### Basic Algorithm
- **Instance-based learning**: No explicit training phase
- **Lazy learning**: All computation happens at prediction time
- **Distance-based**: Classification based on nearest neighbors
- **Majority voting**: Most common class among k nearest neighbors

#### KNN Classification Process
1. **Store training data**: Keep all training examples in memory
2. **Calculate distances**: Compute distance from test point to all training points
3. **Find k nearest**: Identify k closest training points
4. **Majority vote**: Assign class that appears most frequently among k neighbors
5. **Optional weighting**: Weight votes by distance (closer points have more influence)

### Distance Metrics

#### Common Distance Measures
- **Euclidean distance**: Straight-line distance in Euclidean space
- **Manhattan distance**: Sum of absolute differences
- **Minkowski distance**: Generalized distance metric
- **Hamming distance**: For categorical/binary data
- **Cosine similarity**: Angle-based similarity measure

#### Euclidean Distance Formula
```
d(p,q) = √∑(p_i - q_i)²
```

#### Manhattan Distance Formula
```
d(p,q) = ∑|p_i - q_i|
```

### KNN Advantages and Limitations

#### Advantages
- **Simple and intuitive**: Easy to understand and implement
- **No training phase**: Fast to "train" (just store data)
- **Non-parametric**: No assumptions about data distribution
- **Versatile**: Works for classification and regression
- **Handles multi-class**: Naturally extends to multiple classes

#### Limitations
- **Computational cost**: Expensive prediction (distance to all points)
- **Memory intensive**: Must store entire training dataset
- **Curse of dimensionality**: Performance degrades in high dimensions
- **Sensitive to scale**: Features must be normalized
- **Class imbalance**: Can be biased towards majority class

## GPU Acceleration with RAPIDS

### Why GPU for KNN?

#### Computational Characteristics
- **Distance calculations**: Highly parallelizable matrix operations
- **Memory bandwidth**: GPUs excel at data-parallel computations
- **Large datasets**: Can handle datasets that don't fit in CPU memory
- **Batch processing**: Efficient processing of multiple queries

#### RAPIDS cuML Implementation
- **cuML KNN**: GPU-accelerated k-nearest neighbors
- **Multiple distance metrics**: Euclidean, Manhattan, etc.
- **Batch processing**: Handle multiple queries simultaneously
- **Memory efficient**: Optimized GPU memory usage

### Basic GPU KNN Implementation
```python
import cudf
import cupy as cp
from cuml.neighbors import KNeighborsClassifier
from cuml.metrics import accuracy_score

# Load data into GPU memory
X_train_gpu = cudf.DataFrame.from_pandas(X_train)
y_train_gpu = cudf.Series(y_train)
X_test_gpu = cudf.DataFrame.from_pandas(X_test)
y_test_gpu = cudf.Series(y_test)

# Create GPU KNN classifier
gpu_knn = KNeighborsClassifier(
    n_neighbors=5,              # Number of neighbors
    metric='euclidean',         # Distance metric
    algorithm='brute',          # Algorithm (brute force on GPU)
    n_jobs=None                 # Not used for GPU
)

# Fit the model (just stores the data)
gpu_knn.fit(X_train_gpu, y_train_gpu)

# Make predictions
y_pred_gpu = gpu_knn.predict(X_test_gpu)

# Evaluate
accuracy = accuracy_score(y_test_gpu, y_pred_gpu)
print(f"GPU KNN Accuracy: {accuracy:.3f}")
```

### KNN with Different Distance Metrics
```python
# Different distance metrics
metrics = ['euclidean', 'manhattan', 'minkowski']

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_gpu, y_train_gpu)
    y_pred = knn.predict(X_test_gpu)
    acc = accuracy_score(y_test_gpu, y_pred)
    print(f"{metric.capitalize()} distance: {acc:.3f}")
```

## Performance Comparison

### CPU vs GPU Benchmarking
```python
import time
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from cuml.neighbors import KNeighborsClassifier as CumlKNN

def benchmark_knn(X_train, y_train, X_test, y_test, k_values):
    """
    Compare CPU and GPU KNN performance
    """
    results = []
    
    for k in k_values:
        print(f"Testing with k={k}...")
        
        # CPU KNN
        cpu_start = time.time()
        cpu_knn = SklearnKNN(n_neighbors=k, algorithm='brute', n_jobs=-1)
        cpu_knn.fit(X_train, y_train)
        cpu_pred = cpu_knn.predict(X_test)
        cpu_time = time.time() - cpu_start
        cpu_accuracy = accuracy_score(y_test, cpu_pred)
        
        # GPU KNN
        gpu_start = time.time()
        gpu_knn = CumlKNN(n_neighbors=k)
        gpu_knn.fit(X_train_gpu, y_train_gpu)
        gpu_pred = gpu_knn.predict(X_test_gpu)
        gpu_time = time.time() - gpu_start
        gpu_accuracy = accuracy_score(y_test_gpu, gpu_pred)
        
        results.append({
            'k': k,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time / gpu_time if gpu_time > 0 else float('inf'),
            'cpu_accuracy': cpu_accuracy,
            'gpu_accuracy': gpu_accuracy
        })
    
    return pd.DataFrame(results)

# Run benchmark
k_values = [3, 5, 7, 10, 15]
benchmark_results = benchmark_knn(X_train, y_train, X_test, y_test, k_values)

print(benchmark_results)

# Visualize results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(benchmark_results['k'], benchmark_results['cpu_time'], 'b-o', label='CPU')
plt.plot(benchmark_results['k'], benchmark_results['gpu_time'], 'r-o', label='GPU')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Prediction Time (seconds)')
plt.title('Prediction Time Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(benchmark_results['k'], benchmark_results['speedup'], 'g-o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Speedup (CPU time / GPU time)')
plt.title('GPU Speedup')
plt.grid(True)

plt.tight_layout()
plt.show()
```

#### Expected Performance Gains
- **Small datasets**: 2-5x speedup
- **Medium datasets**: 5-20x speedup
- **Large datasets**: 20-100x speedup
- **Batch prediction**: Even larger speedups for multiple queries

## Hyperparameter Tuning

### Choosing Optimal k
```python
# Test different k values
k_values = range(1, 21, 2)
train_scores = []
test_scores = []

for k in k_values:
    knn = CumlKNN(n_neighbors=k)
    knn.fit(X_train_gpu, y_train_gpu)
    
    # Training accuracy (on subset for speed)
    train_pred = knn.predict(X_train_gpu[:1000])
    train_acc = accuracy_score(y_train_gpu[:1000], train_pred)
    train_scores.append(train_acc)
    
    # Test accuracy
    test_pred = knn.predict(X_test_gpu)
    test_acc = accuracy_score(y_test_gpu, test_pred)
    test_scores.append(test_acc)

# Plot k vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, 'b-', label='Training Accuracy')
plt.plot(k_values, test_scores, 'r-', label='Test Accuracy')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: k vs Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Find optimal k
optimal_k = k_values[np.argmax(test_scores)]
print(f"Optimal k: {optimal_k}")
```

### Distance Metric Selection
```python
# Compare different distance metrics
metrics = ['euclidean', 'manhattan', 'cosine', 'minkowski']
metric_scores = []

for metric in metrics:
    try:
        knn = CumlKNN(n_neighbors=optimal_k, metric=metric)
        knn.fit(X_train_gpu, y_train_gpu)
        y_pred = knn.predict(X_test_gpu)
        acc = accuracy_score(y_test_gpu, y_pred)
        metric_scores.append(acc)
        print(f"{metric}: {acc:.3f}")
    except:
        print(f"{metric}: Not supported")
        metric_scores.append(0)

# Plot metric comparison
plt.figure(figsize=(8, 5))
plt.bar(metrics, metric_scores)
plt.xlabel('Distance Metric')
plt.ylabel('Accuracy')
plt.title('KNN: Distance Metric Comparison')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.show()
```

## Handling Large Datasets

### Approximate KNN for Speed
```python
# For very large datasets, consider approximate methods
# Note: cuML doesn't have built-in approximate KNN
# You might need to use other libraries or implement sampling

def approximate_knn_predictions(X_train, y_train, X_test, sample_size=10000):
    """
    Approximate KNN by sampling training data
    """
    # Randomly sample training data
    indices = np.random.choice(len(X_train), size=sample_size, replace=False)
    X_sample = X_train[indices]
    y_sample = y_train[indices]
    
    # Convert to GPU
    X_sample_gpu = cudf.DataFrame.from_pandas(X_sample)
    y_sample_gpu = cudf.Series(y_sample)
    X_test_gpu = cudf.DataFrame.from_pandas(X_test)
    
    # Train on sample
    knn_approx = CumlKNN(n_neighbors=5)
    knn_approx.fit(X_sample_gpu, y_sample_gpu)
    
    # Predict
    y_pred = knn_approx.predict(X_test_gpu)
    
    return y_pred
```

### Batch Processing for Multiple Queries
```python
# Process multiple test samples efficiently
def batch_knn_predictions(knn_model, X_test_batch, batch_size=1000):
    """
    Process predictions in batches to manage memory
    """
    predictions = []
    
    for i in range(0, len(X_test_batch), batch_size):
        batch_end = min(i + batch_size, len(X_test_batch))
        X_batch = X_test_batch[i:batch_end]
        
        # Convert batch to GPU
        X_batch_gpu = cudf.DataFrame.from_pandas(X_batch)
        
        # Predict batch
        batch_pred = knn_model.predict(X_batch_gpu)
        predictions.extend(batch_pred.values_host)
    
    return np.array(predictions)
```

## KNN for Different Data Types

### Categorical Data Handling
```python
# For categorical features, use appropriate distance metrics
# One-hot encode categorical variables first
from sklearn.preprocessing import OneHotEncoder

# Example with mixed data types
categorical_features = ['color', 'size', 'brand']
numerical_features = ['price', 'weight', 'rating']

# One-hot encode categorical
encoder = OneHotEncoder(sparse=False)
cat_encoded = encoder.fit_transform(X[categorical_features])

# Combine with numerical
X_processed = np.concatenate([X[numerical_features].values, cat_encoded], axis=1)

# Use Hamming distance for binary/categorical, Euclidean for numerical
# Note: cuML KNN supports various metrics but may need preprocessing
```

### Text Data with KNN
```python
# For text classification with KNN
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_text_tfidf = vectorizer.fit_transform(text_data)

# Convert to dense array for cuML
X_text_dense = X_text_tfidf.toarray()

# Use cosine distance for text
X_text_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_text_dense))
knn_text = CumlKNN(n_neighbors=5, metric='cosine')
knn_text.fit(X_text_gpu, y_gpu)
```

## Practical Implementation

### Complete GPU KNN Pipeline
```python
import pandas as pd
import numpy as np
import cudf
import cupy as cp
from cuml.neighbors import KNeighborsClassifier
from cuml.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time

# Load and prepare data
print("Loading data...")
df = pd.read_csv('dataset.csv')

# Handle categorical variables if any
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("Encoding categorical variables...")
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

X = df.drop('target', axis=1)
y = df['target']

# Preprocessing
print("Preprocessing...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples")

# Convert to GPU dataframes
print("Converting to GPU dataframes...")
X_train_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
y_train_gpu = cudf.Series(y_train.values)
X_test_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_test))
y_test_gpu = cudf.Series(y_test.values)

# GPU KNN training and evaluation
print("Training and evaluating GPU KNN...")

# Test different k values
k_values = [1, 3, 5, 7, 9, 11, 15]
knn_results = []

for k in k_values:
    start_time = time.time()
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_gpu, y_train_gpu)
    
    y_pred = knn.predict(X_test_gpu)
    accuracy = accuracy_score(y_test_gpu, y_pred)
    
    elapsed_time = time.time() - start_time
    
    knn_results.append({
        'k': k,
        'accuracy': accuracy,
        'time': elapsed_time
    })
    
    print(f"k={k}: Accuracy={accuracy:.3f}, Time={elapsed_time:.3f}s")

# Find best k
best_result = max(knn_results, key=lambda x: x['accuracy'])
best_k = best_result['k']
best_accuracy = best_result['accuracy']

print(f"\nBest k: {best_k}, Best accuracy: {best_accuracy:.3f}")

# Train final model with best k
print("Training final model...")
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_gpu, y_train_gpu)

# Comprehensive evaluation
print("Final model evaluation...")
y_pred_final = final_knn.predict(X_test_gpu)
y_prob_final = final_knn.predict_proba(X_test_gpu)

final_accuracy = accuracy_score(y_test_gpu, y_pred_final)

print(f"\n=== Final Model Results ===")
print(f"Accuracy: {final_accuracy:.3f}")

# Classification report
print("\n=== Classification Report ===")
# Convert to CPU for sklearn metrics
y_pred_cpu = y_pred_final.values_host
y_test_cpu = y_test_gpu.values_host
print(classification_report(y_test_cpu, y_pred_cpu))

# Confusion matrix
cm = confusion_matrix(y_test_gpu, y_pred_final)
print("\n=== Confusion Matrix ===")
print(cm)

# Visualize k vs accuracy
plt.figure(figsize=(10, 6))
k_list = [r['k'] for r in knn_results]
acc_list = [r['accuracy'] for r in knn_results]
time_list = [r['time'] for r in knn_results]

plt.subplot(1, 2, 1)
plt.plot(k_list, acc_list, 'b-o', linewidth=2, markersize=6)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: k vs Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_list, time_list, 'r-o', linewidth=2, markersize=6)
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Prediction Time (seconds)')
plt.title('KNN: k vs Prediction Time')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare with CPU version (optional)
try:
    print("Comparing with CPU KNN...")
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    
    cpu_knn = SklearnKNN(n_neighbors=best_k, n_jobs=-1)
    
    start_time = time.time()
    cpu_knn.fit(X_train, y_train)
    cpu_pred = cpu_knn.predict(X_test)
    cpu_time = time.time() - start_time
    
    cpu_accuracy = accuracy_score(y_test, cpu_pred)
    
    print(f"CPU Prediction time: {cpu_time:.3f} seconds")
    print(f"GPU Prediction time: {best_result['time']:.3f} seconds")
    print(f"Speedup: {cpu_time/best_result['time']:.1f}x")
    print(f"CPU Accuracy: {cpu_accuracy:.3f}")
    print(f"GPU Accuracy: {best_accuracy:.3f}")
    
except ImportError:
    print("Scikit-learn not available for comparison")

# Distance metric comparison
print("\nComparing distance metrics...")
metrics = ['euclidean', 'manhattan', 'minkowski']
metric_results = []

for metric in metrics:
    try:
        knn_metric = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
        knn_metric.fit(X_train_gpu, y_train_gpu)
        y_pred_metric = knn_metric.predict(X_test_gpu)
        acc_metric = accuracy_score(y_test_gpu, y_pred_metric)
        metric_results.append(acc_metric)
        print(f"{metric.capitalize()}: {acc_metric:.3f}")
    except Exception as e:
        print(f"{metric.capitalize()}: Error - {e}")
        metric_results.append(0)

# Visualize metric comparison
plt.figure(figsize=(8, 5))
plt.bar(metrics, metric_results)
plt.xlabel('Distance Metric')
plt.ylabel('Accuracy')
plt.title('KNN: Distance Metric Comparison')
plt.grid(True, axis='y')
for i, v in enumerate(metric_results):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
plt.show()

print("\nGPU KNN analysis completed!")
```

## Best Practices

### KNN Optimization
1. **Feature scaling**: Essential for distance-based algorithms
2. **Optimal k selection**: Use cross-validation to find best k
3. **Distance metric**: Choose appropriate metric for your data
4. **Dimensionality reduction**: Consider PCA for high-dimensional data
5. **Data structure**: Use efficient data structures for large datasets

### GPU-Specific Considerations
1. **Memory management**: Monitor GPU memory usage
2. **Batch processing**: Process large datasets in batches
3. **Data transfer**: Minimize CPU-GPU transfers
4. **Precision**: Consider float32 for speed vs float64 for accuracy
5. **Multiple GPUs**: Utilize multiple GPUs when available

### When to Use KNN
- **Small to medium datasets**: Where training time isn't critical
- **Non-linear decision boundaries**: KNN can capture complex patterns
- **Interpretability needed**: Easy to understand predictions
- **Baseline model**: Good starting point for classification tasks
- **Multi-class problems**: Naturally handles multiple classes

## Troubleshooting

### Common KNN Issues
- **Slow predictions**: Use approximate methods or reduce dataset size
- **Memory errors**: Process in batches or use sampling
- **Poor performance**: Check feature scaling and k selection
- **High dimensions**: Consider dimensionality reduction

### GPU-Specific Issues
- **Out of memory**: Reduce batch size or use CPU fallback
- **Data transfer slow**: Keep data on GPU when possible
- **Incompatible metrics**: Check supported distance metrics in cuML

## Connection to Scalable Computing

### Integration with Previous Modules
- **Dask**: Distributed KNN for massive datasets
- **Spark**: KNN within Spark ML pipelines
- **HBase**: Fast neighbor retrieval for large datasets

### Production Considerations
- **Pre-computed neighbors**: Cache frequent queries
- **Approximate methods**: Use for real-time applications
- **Batch processing**: Optimize for production workloads
- **Model updates**: Handle streaming data updates

## Summary

This lecture completes the Module 14 coverage of machine learning classification algorithms with GPU acceleration. We've explored:

1. **Linear Models**: Logistic regression with RAPIDS
2. **Decision Trees**: Basic trees and Random Forest
3. **Ensemble Methods**: Bagging and boosting (XGBoost)
4. **Instance-Based**: KNN with GPU acceleration

Key takeaways:
- GPU acceleration provides significant speedups for ML algorithms
- Ensemble methods generally outperform single models
- Proper evaluation metrics are crucial for model assessment
- Scalable computing integrates seamlessly with ML workflows

The module builds on the distributed computing foundations from Modules 11-13, showing how to scale ML training and inference across clusters and GPUs.