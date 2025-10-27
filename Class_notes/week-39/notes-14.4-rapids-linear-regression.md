# Lecture 14.4: RAPIDS Acceleration for Linear Regression

## Key Learning Objectives
- Understand GPU acceleration benefits for machine learning
- Learn RAPIDS cuML library for accelerated linear models
- Compare CPU vs GPU performance for linear regression
- Implement scalable linear models with RAPIDS

## Core Concepts

### Why GPU Acceleration Matters

#### CPU vs GPU Architecture
- **CPU**: Few cores optimized for sequential processing
- **GPU**: Thousands of cores optimized for parallel processing
- **ML Benefit**: Matrix operations are highly parallelizable

#### Performance Gains
- **Training Speed**: 10-100x faster on GPUs
- **Memory Bandwidth**: GPUs have much higher memory bandwidth
- **Scalability**: Handle larger datasets and models

### RAPIDS Ecosystem Overview

#### Core Libraries
- **cuDF**: GPU DataFrame operations (pandas-like)
- **cuML**: GPU Machine Learning algorithms (scikit-learn-like)
- **cuGraph**: GPU Graph analytics
- **cuSpatial**: GPU Spatial analytics
- **cuCIM**: GPU Computer vision and image processing

#### Key Advantages
- **Drop-in replacement**: Similar API to scikit-learn
- **GPU memory management**: Automatic memory handling
- **Multi-GPU support**: Scale across multiple GPUs
- **Integration**: Works with Dask for distributed computing

## Linear Models in RAPIDS cuML

### Logistic Regression with cuML

#### Basic Implementation
```python
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score

# Create GPU model
gpu_model = LogisticRegression()

# Fit on GPU (data must be on GPU)
gpu_model.fit(X_train_gpu, y_train_gpu)

# Predict
y_pred_gpu = gpu_model.predict(X_test_gpu)

# Evaluate
accuracy = accuracy_score(y_test_gpu, y_pred_gpu)
```

#### Data Transfer to GPU
```python
import cudf
import cupy as cp

# Convert pandas to cudf (GPU DataFrame)
X_train_gpu = cudf.DataFrame.from_pandas(X_train)
y_train_gpu = cudf.Series(y_train)

# Or convert to cupy arrays
X_train_gpu = cp.asarray(X_train)
y_train_gpu = cp.asarray(y_train)
```

### Multi-Class Classification

#### One-vs-Rest with cuML
```python
# Multi-class logistic regression
multi_gpu_model = LogisticRegression(multi_class='ovr')
multi_gpu_model.fit(X_train_gpu, y_train_gpu)
```

#### Softmax Regression
```python
# Multinomial logistic regression
softmax_gpu_model = LogisticRegression(multi_class='multinomial')
softmax_gpu_model.fit(X_train_gpu, y_train_gpu)
```

## Performance Comparison

### CPU vs GPU Benchmarking

#### Setup Comparison Code
```python
import time
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from cuml.linear_model import LogisticRegression as CumlLogisticRegression

# CPU implementation
start_time = time.time()
cpu_model = SklearnLogisticRegression(random_state=42)
cpu_model.fit(X_train, y_train)
cpu_pred = cpu_model.predict(X_test)
cpu_time = time.time() - start_time

# GPU implementation
start_time = time.time()
gpu_model = CumlLogisticRegression()
gpu_model.fit(X_train_gpu, y_train_gpu)
gpu_pred = gpu_model.predict(X_test_gpu)
gpu_time = time.time() - start_time

print(f"CPU time: {cpu_time:.3f}s")
print(f"GPU time: {gpu_time:.3f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

#### Expected Performance Gains
- **Small datasets**: 2-5x speedup
- **Medium datasets**: 10-20x speedup
- **Large datasets**: 50-100x speedup
- **Very large datasets**: Limited by GPU memory

## Regularization and Hyperparameter Tuning

### Regularization Options
```python
# L2 regularization (default)
ridge_gpu = CumlLogisticRegression(penalty='l2', C=1.0)

# L1 regularization
lasso_gpu = CumlLogisticRegression(penalty='l1', C=1.0)

# Elastic net
elastic_gpu = CumlLogisticRegression(penalty='elasticnet', l1_ratio=0.5)
```

### GPU-Accelerated Grid Search
```python
from cuml.model_selection import GridSearchCV
from cuml.linear_model import LogisticRegression

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'max_iter': [100, 200, 500]
}

gpu_grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

gpu_grid.fit(X_train_gpu, y_train_gpu)
print(f"Best params: {gpu_grid.best_params_}")
```

## Memory Management

### GPU Memory Considerations
- **GPU Memory Limits**: Typically 8-24GB on modern GPUs
- **Data Size Limits**: Must fit in GPU memory for training
- **Memory Transfer**: CPU↔GPU transfers are expensive

### Memory Optimization Strategies
```python
# Check GPU memory
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU Memory: {info.used/1024**3:.1f}GB used / {info.total/1024**3:.1f}GB total")
```

### Large Dataset Handling
- **Data chunking**: Process data in batches
- **Dask integration**: Use Dask for out-of-core processing
- **Memory mapping**: Use memory-mapped arrays

## Integration with Dask

### Distributed GPU Training
```python
import dask_cudf
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
from cuml.dask.linear_model import LogisticRegression as DaskCumlLogisticRegression

# Load data with dask-cudf
df = dask_cudf.read_csv('large_dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# Distributed GPU training
dask_gpu_model = DaskCumlLogisticRegression()
dask_gpu_model.fit(X, y)
```

### Multi-GPU Scaling
- **Automatic distribution**: Dask handles GPU assignment
- **Load balancing**: Work distributed across available GPUs
- **Fault tolerance**: Handles GPU failures gracefully

## Advanced Features

### Custom Solvers
```python
# Different solvers for different problems
gpu_model_qn = CumlLogisticRegression(solver='qn')  # Quasi-Newton
gpu_model_lbfgs = CumlLogisticRegression(solver='lbfgs')  # Limited-memory BFGS
gpu_model_sgd = CumlLogisticRegression(solver='sgd')  # Stochastic Gradient Descent
```

### Convergence Monitoring
```python
# Monitor convergence
gpu_model = CumlLogisticRegression(verbose=True, max_iter=1000, tol=1e-6)
gpu_model.fit(X_train_gpu, y_train_gpu)

# Check convergence
print(f"Converged: {gpu_model.converged_}")
print(f"Iterations: {gpu_model.n_iter_}")
```

## Practical Implementation

### Complete GPU Workflow
```python
import cudf
import cuml
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score, classification_report
from cuml.preprocessing import StandardScaler

# Load and preprocess data
df = cudf.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = cuml.train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

## Performance Best Practices

### When to Use GPU Acceleration
- **Large datasets**: > 100K samples
- **High-dimensional data**: > 100 features
- **Iterative algorithms**: Many model evaluations
- **Real-time prediction**: Low latency requirements

### Optimization Tips
1. **Minimize CPU-GPU transfers**: Keep data on GPU
2. **Batch processing**: Process multiple predictions together
3. **Memory monitoring**: Watch GPU memory usage
4. **Algorithm selection**: Choose GPU-optimized algorithms

## Troubleshooting

### Common Issues
- **Out of memory**: Reduce batch size or use CPU fallback
- **Slow transfers**: Minimize data movement between CPU/GPU
- **Convergence issues**: Adjust tolerance and max iterations
- **Installation problems**: Ensure proper CUDA toolkit version

### Debugging GPU Code
```python
# Check GPU availability
print(f"GPU available: {cuml.is_cuda_available()}")

# Check cuML version
print(f"cuML version: {cuml.__version__}")

# Monitor GPU usage
import GPUtil
GPUtil.showUtilization()
```

## Connection to Scalable Computing

### Integration with Previous Modules
- **Spark**: Use GPU acceleration within Spark jobs
- **HBase**: Fast feature retrieval for GPU models
- **Dask**: Distributed GPU computing at scale

### Production Deployment
- **Model serving**: Deploy GPU models with Triton Inference Server
- **Auto-scaling**: Scale GPU resources based on demand
- **Cost optimization**: Balance performance vs cost

## Next Steps

This lecture demonstrates GPU acceleration for linear models. The next lecture (14.5) will cover overfitting prevention and cross-validation techniques, essential for building robust machine learning models.