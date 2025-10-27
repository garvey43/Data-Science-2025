# Lecture 14.12: XGBoost with RAPIDS Acceleration

## Key Learning Objectives
- Understand GPU acceleration for XGBoost
- Learn RAPIDS XGBoost implementation
- Compare CPU vs GPU XGBoost performance
- Implement distributed XGBoost with Dask

## Core Concepts

### XGBoost GPU Acceleration

#### Why GPU for XGBoost?
- **Parallel tree construction**: Multiple trees built simultaneously
- **Histogram computation**: Fast parallel histogram building
- **Memory bandwidth**: High-speed GPU memory access
- **Scalability**: Handle larger datasets and more trees

#### GPU vs CPU XGBoost
- **Algorithm differences**: GPU uses different tree building strategies
- **Memory layout**: Optimized for GPU architecture
- **Parallel processing**: Better utilization of GPU cores
- **Performance gains**: 5-20x speedup depending on dataset size

### RAPIDS XGBoost Architecture

#### Core Components
- **GPU Histograms**: Fast parallel histogram computation
- **Memory Management**: Efficient GPU memory usage
- **Tree Building**: GPU-optimized tree construction algorithms
- **Data Handling**: cuDF integration for seamless data transfer

#### Key Optimizations
- **External Memory**: Handle datasets larger than GPU memory
- **Multi-GPU Support**: Scale across multiple GPUs
- **Compressed Storage**: Efficient data compression
- **Asynchronous Processing**: Overlap computation and data transfer

## Implementation

### Basic GPU XGBoost
```python
import cudf
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data into GPU DataFrames
X_train_gpu = cudf.DataFrame.from_pandas(X_train)
y_train_gpu = cudf.Series(y_train)
X_test_gpu = cudf.DataFrame.from_pandas(X_test)
y_test_gpu = cudf.Series(y_test)

# Create DMatrix for GPU
dtrain = xgb.DMatrix(X_train_gpu, label=y_train_gpu)
dtest = xgb.DMatrix(X_test_gpu, label=y_test_gpu)

# Set GPU parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist',    # Use GPU histogram method
    'gpu_id': 0,                  # GPU device ID
    'predictor': 'gpu_predictor'  # Use GPU for prediction
}

# Train on GPU
num_rounds = 100
evals_result = {}
bst = xgb.train(
    params, 
    dtrain, 
    num_rounds, 
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    evals_result=evals_result,
    verbose_eval=10
)

# Make predictions
y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"GPU XGBoost Accuracy: {accuracy:.3f}")
print(f"GPU XGBoost AUC: {auc:.3f}")
```

### Using XGBoost Python API with GPU
```python
from xgboost import XGBClassifier

# GPU XGBoost classifier
gpu_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='gpu_hist',       # GPU histogram
    gpu_id=0,                     # GPU device
    predictor='gpu_predictor',    # GPU prediction
    random_state=42
)

# Train on GPU
gpu_xgb.fit(X_train, y_train)

# Predictions
y_pred = gpu_xgb.predict(X_test)
y_prob = gpu_xgb.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_prob[:, 1]):.3f}")
```

## Performance Comparison

### CPU vs GPU Benchmarking
```python
import time
from xgboost import XGBClassifier

def benchmark_xgboost(X_train, y_train, X_test, y_test, n_estimators_list):
    """
    Compare CPU and GPU XGBoost performance
    """
    results = []
    
    for n_est in n_estimators_list:
        print(f"Testing with {n_est} estimators...")
        
        # CPU XGBoost
        cpu_start = time.time()
        cpu_xgb = XGBClassifier(
            n_estimators=n_est,
            tree_method='hist',  # CPU histogram
            random_state=42
        )
        cpu_xgb.fit(X_train, y_train)
        cpu_pred = cpu_xgb.predict(X_test)
        cpu_time = time.time() - cpu_start
        cpu_accuracy = accuracy_score(y_test, cpu_pred)
        
        # GPU XGBoost
        gpu_start = time.time()
        gpu_xgb = XGBClassifier(
            n_estimators=n_est,
            tree_method='gpu_hist',    # GPU histogram
            gpu_id=0,
            predictor='gpu_predictor',
            random_state=42
        )
        gpu_xgb.fit(X_train, y_train)
        gpu_pred = gpu_xgb.predict(X_test)
        gpu_time = time.time() - gpu_start
        gpu_accuracy = accuracy_score(y_test, gpu_pred)
        
        results.append({
            'n_estimators': n_est,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time / gpu_time,
            'cpu_accuracy': cpu_accuracy,
            'gpu_accuracy': gpu_accuracy
        })
    
    return pd.DataFrame(results)

# Run benchmark
n_estimators_list = [50, 100, 200, 500]
benchmark_results = benchmark_xgboost(X_train, y_train, X_test, y_test, n_estimators_list)

print(benchmark_results)

# Visualize results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(benchmark_results['n_estimators'], benchmark_results['cpu_time'], 'b-o', label='CPU')
plt.plot(benchmark_results['n_estimators'], benchmark_results['gpu_time'], 'r-o', label='GPU')
plt.xlabel('Number of Estimators')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(benchmark_results['n_estimators'], benchmark_results['speedup'], 'g-o')
plt.xlabel('Number of Estimators')
plt.ylabel('Speedup (CPU time / GPU time)')
plt.title('GPU Speedup')
plt.grid(True)

plt.tight_layout()
plt.show()
```

#### Expected Performance Gains
- **Small datasets (< 10K samples)**: 2-5x speedup
- **Medium datasets (10K-100K samples)**: 5-15x speedup
- **Large datasets (> 100K samples)**: 10-30x speedup
- **Very large datasets**: Limited by GPU memory

## Advanced GPU Features

### Multi-GPU Training
```python
# Set up multi-GPU training
params = {
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'gpu_id': 0,  # Starting GPU
    'updater': 'grow_gpu_hist',  # Multi-GPU updater
    'n_gpus': 2,  # Number of GPUs to use
    'max_depth': 6,
    'learning_rate': 0.1
}

# Train across multiple GPUs
bst_multi = xgb.train(params, dtrain, num_rounds, evals=[(dtest, 'test')])
```

### External Memory Training
```python
# For datasets larger than GPU memory
params = {
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'max_depth': 6,
    'extmem_single_page': True,  # Enable external memory
    'max_bin': 256,              # Reduce memory usage
    'gpu_id': 0
}

# Create DMatrix with external memory
dtrain_ext = xgb.DMatrix('large_train.csv?format=csv&label_column=0')
bst_ext = xgb.train(params, dtrain_ext, num_rounds)
```

### GPU Memory Optimization
```python
# Monitor GPU memory usage
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def print_gpu_memory():
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU Memory: {info.used/1024**3:.1f}GB used / {info.total/1024**3:.1f}GB total")

print("Before training:")
print_gpu_memory()

# Training with memory monitoring
bst = xgb.train(params, dtrain, num_rounds)

print("After training:")
print_gpu_memory()
```

## Distributed XGBoost with Dask

### Dask XGBoost Integration
```python
import dask
import dask_cudf
from dask.distributed import Client, LocalCluster
import xgboost as xgb
import dask_xgboost as dxgb

# Start Dask cluster
cluster = LocalCluster(n_workers=2, threads_per_worker=1)
client = Client(cluster)

# Load data with dask-cudf
X_dask = dask_cudf.read_csv('large_dataset.csv', usecols=X_columns)
y_dask = dask_cudf.read_csv('large_dataset.csv', usecols=['target'])

# Distributed GPU training
params = {
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'max_depth': 6,
    'learning_rate': 0.1
}

# Train distributed XGBoost
bst_distributed = dxgb.train(
    client, 
    params, 
    X_dask, 
    y_dask, 
    num_boost_round=100
)

# Make distributed predictions
y_pred_distributed = dxgb.predict(client, bst_distributed, X_dask)
```

### RAPIDS cuML XGBoost Alternative
```python
# Using RAPIDS cuML for XGBoost-like functionality
from cuml import ForestInference

# Load trained XGBoost model into RAPIDS
# (Note: This requires converting XGBoost model to Treelite format)
# This is more complex and typically done for inference optimization

# For training, stick with XGBoost GPU implementation
```

## Hyperparameter Tuning on GPU

### GPU-Accelerated Grid Search
```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Define parameter grid
param_grid = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# GPU XGBoost for grid search
gpu_xgb = XGBClassifier(
    tree_method='gpu_hist',
    gpu_id=0,
    predictor='gpu_predictor',
    random_state=42
)

# Grid search with GPU acceleration
grid_search = GridSearchCV(
    gpu_xgb,
    param_grid,
    cv=3,  # Fewer folds for speed
    scoring='accuracy',
    n_jobs=1,  # GPU doesn't benefit from multiple jobs
    verbose=2
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

## Model Interpretation on GPU

### GPU Feature Importance
```python
# Get feature importance from GPU model
feature_importance = gpu_xgb.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 Important Features:")
print(importance_df.head(10))

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
plt.xlabel('Feature Importance')
plt.title('GPU XGBoost Feature Importance')
plt.gca().invert_yaxis()
plt.show()
```

### SHAP Values with GPU
```python
import shap

# Use GPU for SHAP computation (if available)
# Note: SHAP GPU support is limited, typically use CPU for interpretation

# For large datasets, sample for SHAP
X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(gpu_xgb)
shap_values = explainer.shap_values(X_sample)

# Summary plot
shap.summary_plot(shap_values, X_sample, feature_names=X.columns)
```

## Practical Implementation

### Complete GPU XGBoost Pipeline
```python
import pandas as pd
import numpy as np
import cudf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# Load and prepare data
print("Loading data...")
df = pd.read_csv('large_dataset.csv')
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

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Convert to GPU dataframes for cuML compatibility
print("Converting to GPU dataframes...")
X_train_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_train, columns=X.columns))
y_train_gpu = cudf.Series(y_train.values)
X_test_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_test, columns=X.columns))
y_test_gpu = cudf.Series(y_test.values)

# GPU XGBoost training
print("Training GPU XGBoost...")
start_time = time.time()

# Using DMatrix for GPU training
dtrain = xgb.DMatrix(X_train_gpu, label=y_train_gpu)
dtest = xgb.DMatrix(X_test_gpu, label=y_test_gpu)

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'random_state': 42
}

evals_result = {}
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=20,
    evals_result=evals_result,
    verbose_eval=50
)

gpu_training_time = time.time() - start_time
print(f"GPU training completed in {gpu_training_time:.2f} seconds")

# Evaluate on GPU
print("Evaluating model...")
y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

gpu_accuracy = accuracy_score(y_test, y_pred)
gpu_auc = roc_auc_score(y_test, y_pred_prob)

print(f"GPU XGBoost Accuracy: {gpu_accuracy:.3f}")
print(f"GPU XGBoost AUC: {gpu_auc:.3f}")
print(f"Best iteration: {bst.best_iteration}")

# Classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n=== Confusion Matrix ===")
print(cm)

# Feature importance
print("\nAnalyzing feature importance...")
importance_dict = bst.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': list(importance_dict.keys()),
    'importance': list(importance_dict.values())
}).sort_values('importance', ascending=False)

print("Top 10 Important Features:")
print(importance_df.head(10))

# Visualize training curves
epochs = len(evals_result['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x_axis, evals_result['validation_0']['logloss'], label='Train')
plt.plot(x_axis, evals_result['validation_1']['logloss'], label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('GPU XGBoost Log Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_axis, evals_result['validation_0']['auc'], label='Train')
plt.plot(x_axis, evals_result['validation_1']['auc'], label='Test')
plt.xlabel('Boosting Round')
plt.ylabel('AUC')
plt.title('GPU XGBoost AUC')
plt.legend()

plt.tight_layout()
plt.show()

# Compare with CPU version (optional)
try:
    print("Comparing with CPU XGBoost...")
    cpu_params = params.copy()
    cpu_params['tree_method'] = 'hist'  # CPU histogram
    cpu_params.pop('gpu_id', None)
    cpu_params.pop('predictor', None)
    
    start_time = time.time()
    cpu_bst = xgb.train(cpu_params, xgb.DMatrix(X_train, y_train), num_boost_round=50)
    cpu_time = time.time() - start_time
    
    cpu_pred_prob = cpu_bst.predict(xgb.DMatrix(X_test))
    cpu_accuracy = accuracy_score(y_test, (cpu_pred_prob > 0.5).astype(int))
    
    print(f"CPU Training time: {cpu_time:.2f} seconds")
    print(f"GPU Training time: {gpu_training_time:.2f} seconds")
    print(f"Speedup: {cpu_time/gpu_training_time:.1f}x")
    print(f"CPU Accuracy: {cpu_accuracy:.3f}")
    print(f"GPU Accuracy: {gpu_accuracy:.3f}")
    
except Exception as e:
    print(f"CPU comparison failed: {e}")

# Save model
print("Saving model...")
bst.save_model('gpu_xgboost_model.json')
print("Model saved as 'gpu_xgboost_model.json'")

# Load and use saved model
print("Testing model loading...")
loaded_bst = xgb.Booster()
loaded_bst.load_model('gpu_xgboost_model.json')

loaded_pred_prob = loaded_bst.predict(dtest)
loaded_accuracy = accuracy_score(y_test, (loaded_pred_prob > 0.5).astype(int))
print(f"Loaded model accuracy: {loaded_accuracy:.3f}")

print("\nGPU XGBoost training and evaluation completed!")
```

## Best Practices

### GPU XGBoost Optimization
1. **Use appropriate tree_method**: 'gpu_hist' for most cases
2. **Set max_bin carefully**: Balance speed vs accuracy
3. **Monitor GPU memory**: Use external memory for large datasets
4. **Batch predictions**: Process multiple samples together
5. **Early stopping**: Prevent overfitting and save time

### Performance Tuning
1. **GPU memory**: Ensure sufficient GPU memory
2. **Data transfer**: Minimize CPU-GPU transfers
3. **Batch size**: Optimize for GPU architecture
4. **Multiple GPUs**: Use when available
5. **Monitoring**: Track GPU utilization

### Production Deployment
1. **Model serialization**: Save and load GPU models
2. **Batch inference**: Optimize for production workloads
3. **Resource management**: Monitor GPU resources
4. **Fallback options**: Have CPU fallback for edge cases

## Troubleshooting

### Common GPU Issues
- **Out of memory**: Reduce max_depth, use external memory, or subsample data
- **Slow data transfer**: Keep data on GPU, use pinned memory
- **Convergence issues**: Check data preprocessing, adjust learning rate
- **Installation problems**: Ensure CUDA compatibility

### Performance Debugging
```python
# Check GPU utilization
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

# Time different operations
import time

# Time data transfer
start = time.time()
X_gpu = cudf.DataFrame.from_pandas(X)
transfer_time = time.time() - start
print(f"Data transfer time: {transfer_time:.3f} seconds")

# Time training
start = time.time()
bst = xgb.train(params, dtrain, num_rounds)
training_time = time.time() - start
print(f"Training time: {training_time:.3f} seconds")

# Time prediction
start = time.time()
pred = bst.predict(dtest)
pred_time = time.time() - start
print(f"Prediction time: {pred_time:.3f} seconds")
```

## Connection to Scalable Computing

### Integration with Previous Modules
- **Dask**: Distributed GPU XGBoost for massive datasets
- **Spark**: GPU acceleration within Spark ML pipelines
- **HBase**: Fast feature retrieval for GPU models

### Enterprise Applications
- **Real-time scoring**: Low-latency GPU inference
- **Large-scale training**: Distributed GPU training
- **Model serving**: GPU-accelerated model deployment
- **A/B testing**: Fast model comparison and evaluation

## Next Steps

This lecture covers XGBoost with RAPIDS acceleration. The final lecture (14.13) will explore KNN with RAPIDS, completing the coverage of major classification algorithms with GPU acceleration.