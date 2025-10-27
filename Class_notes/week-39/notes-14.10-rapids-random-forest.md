# Lecture 14.10: RAPIDS Acceleration for Random Forest

## Key Learning Objectives
- Understand GPU acceleration benefits for Random Forest
- Learn RAPIDS cuML Random Forest implementation
- Compare CPU vs GPU performance for ensemble methods
- Implement scalable Random Forest with distributed computing

## Core Concepts

### Why GPU Acceleration for Random Forest?

#### Computational Intensity
- **Tree construction**: Many split decisions and calculations
- **Ensemble nature**: Hundreds of trees to train
- **Matrix operations**: Feature selection and evaluation
- **Parallel potential**: Independent tree training

#### GPU Advantages
- **Massive parallelism**: Thousands of CUDA cores
- **High memory bandwidth**: Fast data transfer
- **SIMD operations**: Single Instruction, Multiple Data
- **Scalability**: Multiple GPUs for larger ensembles

### RAPIDS cuML Random Forest

#### Architecture Differences
- **CPU Random Forest**: Sequential tree building, limited parallelism
- **GPU Random Forest**: Parallel tree construction, batched operations
- **Memory layout**: Optimized for GPU memory access patterns
- **Algorithm adaptations**: GPU-specific optimizations

## Implementation with RAPIDS

### Basic GPU Random Forest
```python
import cudf
import cupy as cp
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score, roc_auc_score

# Load data (assuming pandas DataFrame)
X_train_gpu = cudf.DataFrame.from_pandas(X_train)
y_train_gpu = cudf.Series(y_train)
X_test_gpu = cudf.DataFrame.from_pandas(X_test)
y_test_gpu = cudf.Series(y_test)

# Create GPU Random Forest
gpu_rf = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,            # Maximum tree depth
    max_features='sqrt',     # Feature subset size
    bootstrap=True,          # Use bootstrap sampling
    random_state=42,
    n_streams=1              # Number of CUDA streams
)

# Train on GPU
gpu_rf.fit(X_train_gpu, y_train_gpu)

# Make predictions
y_pred_gpu = gpu_rf.predict(X_test_gpu)
y_prob_gpu = gpu_rf.predict_proba(X_test_gpu)

# Evaluate
accuracy = accuracy_score(y_test_gpu, y_pred_gpu)
print(f"GPU Random Forest Accuracy: {accuracy:.3f}")
```

### Key Parameters in cuML

#### GPU-Specific Parameters
```python
gpu_rf_advanced = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_streams=4,             # Multiple CUDA streams for parallelism
    max_batch_size=4096,     # Batch size for processing
    split_criterion='gini',  # Splitting criterion
    min_samples_split=2,
    min_samples_leaf=1
)
```

## Performance Comparison

### CPU vs GPU Benchmarking

#### Comprehensive Benchmark
```python
import time
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from cuml.ensemble import RandomForestClassifier as CumlRF

def benchmark_random_forest(X_train, y_train, X_test, y_test, n_estimators_list):
    """
    Compare CPU and GPU Random Forest performance
    """
    results = []
    
    for n_est in n_estimators_list:
        print(f"Testing with {n_est} estimators...")
        
        # CPU implementation
        cpu_start = time.time()
        cpu_rf = SklearnRF(
            n_estimators=n_est, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        )
        cpu_rf.fit(X_train, y_train)
        cpu_pred = cpu_rf.predict(X_test)
        cpu_time = time.time() - cpu_start
        cpu_accuracy = accuracy_score(y_test, cpu_pred)
        
        # GPU implementation
        gpu_start = time.time()
        gpu_rf = CumlRF(
            n_estimators=n_est, 
            max_depth=10, 
            random_state=42
        )
        gpu_rf.fit(X_train_gpu, y_train_gpu)
        gpu_pred = gpu_rf.predict(X_test_gpu)
        gpu_time = time.time() - gpu_start
        gpu_accuracy = accuracy_score(y_test_gpu, gpu_pred)
        
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
benchmark_results = benchmark_random_forest(
    X_train, y_train, X_test, y_test, n_estimators_list
)

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
- **Large datasets (> 100K samples)**: 20-50x speedup
- **Very large datasets**: Limited by GPU memory

## Memory Management

### GPU Memory Considerations

#### Memory Requirements
- **Model storage**: Each tree requires memory
- **Dataset size**: Must fit in GPU memory
- **Intermediate computations**: Temporary memory for splits
- **Multiple streams**: Additional memory for parallel processing

#### Memory Optimization
```python
# Check GPU memory before training
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(f"GPU Memory: {info.used/1024**3:.1f}GB used / {info.total/1024**3:.1f}GB total")
print(f"Available: {info.free/1024**3:.1f}GB")

# Estimate memory requirements
n_samples, n_features = X_train.shape
estimated_memory = (n_estimators * max_depth * n_features * 4) / 1024**3  # Rough estimate
print(f"Estimated model memory: {estimated_memory:.1f}GB")
```

### Handling Large Datasets

#### Data Chunking
```python
def train_gpu_rf_chunked(X, y, chunk_size=50000, n_estimators=100):
    """
    Train GPU Random Forest on large datasets using chunking
    """
    n_samples = len(X)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    # Initialize with first chunk
    start_idx = 0
    end_idx = min(chunk_size, n_samples)
    
    X_chunk = cudf.DataFrame.from_pandas(X.iloc[start_idx:end_idx])
    y_chunk = cudf.Series(y.iloc[start_idx:end_idx])
    
    gpu_rf = CumlRF(n_estimators=n_estimators//n_chunks, random_state=42)
    gpu_rf.fit(X_chunk, y_chunk)
    
    # Add remaining chunks (simplified - actual implementation would merge trees)
    for i in range(1, n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        
        X_chunk = cudf.DataFrame.from_pandas(X.iloc[start_idx:end_idx])
        y_chunk = cudf.Series(y.iloc[start_idx:end_idx])
        
        # In practice, you'd need to implement tree merging or use Dask
        print(f"Processing chunk {i+1}/{n_chunks}")
    
    return gpu_rf
```

## Feature Importance on GPU

### GPU Feature Importance
```python
# Get feature importance from GPU model
gpu_feature_importance = gpu_rf.feature_importances_

# Convert to pandas for easier handling
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': gpu_feature_importance.values_host if hasattr(gpu_feature_importance, 'values_host') else gpu_feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 Important Features (GPU):")
print(importance_df.head(10))

# Visualize
plt.figure(figsize=(10, 6))
top_features = importance_df.head(15)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.title('GPU Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.show()
```

## Hyperparameter Tuning on GPU

### GPU-Accelerated Grid Search
```python
from sklearn.model_selection import ParameterGrid
import time

def gpu_grid_search(X_train_gpu, y_train_gpu, X_test_gpu, y_test_gpu, param_grid):
    """
    Perform grid search on GPU
    """
    best_score = 0
    best_params = None
    results = []
    
    for params in ParameterGrid(param_grid):
        start_time = time.time()
        
        # Create model with current parameters
        model = CumlRF(random_state=42, **params)
        model.fit(X_train_gpu, y_train_gpu)
        
        # Evaluate
        y_pred = model.predict(X_test_gpu)
        accuracy = accuracy_score(y_test_gpu, y_pred)
        
        elapsed_time = time.time() - start_time
        
        results.append({
            'params': params,
            'accuracy': accuracy,
            'time': elapsed_time
        })
        
        if accuracy > best_score:
            best_score = accuracy
            best_params = params
            
        print(f"Params: {params} -> Accuracy: {accuracy:.3f} (Time: {elapsed_time:.2f}s)")
    
    return best_params, best_score, results

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'max_features': ['sqrt', 'log2']
}

# Run GPU grid search
best_params, best_score, search_results = gpu_grid_search(
    X_train_gpu, y_train_gpu, X_test_gpu, y_test_gpu, param_grid
)

print(f"\nBest parameters: {best_params}")
print(f"Best accuracy: {best_score:.3f}")
```

## Integration with Dask

### Distributed GPU Random Forest
```python
import dask_cudf
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
from cuml.dask.ensemble import RandomForestClassifier as DaskCumlRF

# Load data with dask-cudf for distributed processing
X_dask = dask_cudf.from_pandas(X_train, npartitions=4)
y_dask = dask_cudf.from_pandas(y_train, npartitions=4)

# Distributed GPU Random Forest
dask_gpu_rf = DaskCumlRF(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train on distributed GPU data
dask_gpu_rf.fit(X_dask, y_dask)

# Make predictions
y_pred_dask = dask_gpu_rf.predict(X_dask)

# Compute final results
accuracy = accuracy_score(y_dask.compute(), y_pred_dask.compute())
print(f"Distributed GPU RF Accuracy: {accuracy:.3f}")
```

## Advanced Features

### Multi-GPU Training
```python
# For systems with multiple GPUs
import os

# Set visible GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Use GPUs 0,1,2,3

# cuML will automatically distribute across visible GPUs
multi_gpu_rf = CumlRF(
    n_estimators=500,
    max_depth=15,
    n_streams=4  # One stream per GPU
)

multi_gpu_rf.fit(X_train_gpu, y_train_gpu)
```

### Model Persistence
```python
# Save GPU model
gpu_rf.save('gpu_random_forest.model')

# Load GPU model
loaded_gpu_rf = CumlRF.load('gpu_random_forest.model')

# Make predictions with loaded model
y_pred_loaded = loaded_gpu_rf.predict(X_test_gpu)
```

## Practical Implementation

### Complete GPU Random Forest Pipeline
```python
import pandas as pd
import numpy as np
import cudf
import time
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Convert to GPU dataframes
print("Converting to GPU dataframes...")
X_train_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_train, columns=X.columns))
y_train_gpu = cudf.Series(y_train.values)
X_test_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_test, columns=X.columns))
y_test_gpu = cudf.Series(y_test.values)

# Train GPU Random Forest
print("Training GPU Random Forest...")
start_time = time.time()

gpu_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    max_features='sqrt',
    random_state=42,
    n_streams=2
)

gpu_rf.fit(X_train_gpu, y_train_gpu)
gpu_training_time = time.time() - start_time

print(f"GPU training completed in {gpu_training_time:.2f} seconds")

# Evaluate on GPU
print("Evaluating model...")
y_pred_gpu = gpu_rf.predict(X_test_gpu)
y_prob_gpu = gpu_rf.predict_proba(X_test_gpu)

gpu_accuracy = accuracy_score(y_test_gpu, y_pred_gpu)
gpu_auc = roc_auc_score(y_test_gpu, y_prob_gpu[:, 1])

print(f"GPU Test Accuracy: {gpu_accuracy:.3f}")
print(f"GPU Test AUC: {gpu_auc:.3f}")

# Feature importance analysis
print("Analyzing feature importance...")
feature_importance = gpu_rf.feature_importances_
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance.values_host
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.title('GPU Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Performance comparison with CPU (optional)
try:
    from sklearn.ensemble import RandomForestClassifier as SklearnRF
    
    print("Comparing with CPU implementation...")
    cpu_rf = SklearnRF(
        n_estimators=50,  # Fewer trees for fair comparison
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    cpu_rf.fit(X_train, y_train)
    cpu_training_time = time.time() - start_time
    
    cpu_pred = cpu_rf.predict(X_test)
    cpu_accuracy = accuracy_score(y_test, cpu_pred)
    
    print(f"CPU Training time: {cpu_training_time:.2f} seconds")
    print(f"GPU Training time: {gpu_training_time:.2f} seconds")
    print(f"Speedup: {cpu_training_time/gpu_training_time:.1f}x")
    print(f"CPU Accuracy: {cpu_accuracy:.3f}")
    print(f"GPU Accuracy: {gpu_accuracy:.3f}")
    
except ImportError:
    print("Scikit-learn not available for comparison")

# Model interpretation
print("\nModel Interpretation:")
print(f"Number of trees: {gpu_rf.n_estimators}")
print(f"Max depth: {gpu_rf.max_depth}")
print(f"Max features: {gpu_rf.max_features}")

# Save model
print("Saving model...")
gpu_rf.save('gpu_random_forest_model.model')
print("Model saved successfully!")

print("\nGPU Random Forest training and evaluation completed!")
```

## Best Practices

### GPU Optimization
1. **Memory monitoring**: Track GPU memory usage
2. **Batch processing**: Use appropriate batch sizes
3. **Multiple streams**: Leverage multiple CUDA streams
4. **Data transfer minimization**: Keep data on GPU when possible

### Performance Tuning
1. **Right-size estimators**: Balance accuracy vs speed
2. **Optimal depth**: Tune max_depth for best performance
3. **Feature selection**: Use appropriate max_features
4. **GPU utilization**: Monitor GPU usage with nvidia-smi

### Production Deployment
1. **Model serialization**: Save and load GPU models
2. **Batch prediction**: Process multiple samples efficiently
3. **Resource management**: Monitor GPU resources
4. **Fallback options**: Have CPU fallback for small datasets

## Troubleshooting

### Common GPU Issues
- **Out of memory**: Reduce n_estimators or max_depth, or use data chunking
- **Slow data transfer**: Minimize CPU-GPU transfers
- **Convergence issues**: Check data preprocessing and parameter ranges
- **Installation problems**: Ensure proper CUDA toolkit and RAPIDS versions

### Performance Debugging
```python
# Monitor GPU usage
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

# Check cuML version and GPU info
import cuml
print(f"cuML version: {cuml.__version__}")
print(f"CUDA available: {cuml.is_cuda_available()}")

# Time different operations
import time

start = time.time()
# Operation to time
end = time.time()
print(f"Operation time: {end - start:.3f} seconds")
```

## Connection to Scalable Computing

### Integration with Previous Modules
- **Dask**: Distributed GPU computing for massive datasets
- **Spark**: GPU acceleration within Spark ML pipelines
- **HBase**: Fast feature retrieval for GPU models

### Enterprise Applications
- **Real-time prediction**: Low-latency GPU inference
- **Large-scale training**: Distributed GPU training
- **Model serving**: GPU-accelerated model deployment
- **A/B testing**: Fast model comparison and evaluation

## Next Steps

This lecture demonstrates GPU acceleration for Random Forest. The next lecture (14.11) will introduce boosting methods, another powerful ensemble technique that builds models sequentially to correct previous errors.