# Module 14: Machine Learning Classification - Setup Guide

## Instructor Information
**Lead Instructor**: Dennis Omboga Mongare
**Role**: Lead Data Science Instructor
**Contact**: [Your contact information]
**Course**: Data Science B - Scalable Computing Series

## Environment Setup with UV

This module uses `uv` for dependency management and virtual environment isolation. Follow these steps to set up your environment.

### Prerequisites
- Python 3.13 or higher
- `uv` package manager (install with: `pip install uv` or follow [uv installation guide](https://github.com/astral-sh/uv))
- NVIDIA GPU with CUDA support (for RAPIDS acceleration)

### Setup Commands

#### Linux (Cinnamon Manjaro)
```bash
# 1. Navigate to the module directory
cd "Data-Science-B/14_module_machine_learning_classification"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

#### Windows (PowerShell/Command Prompt)
```powershell
# 1. Navigate to the module directory
cd "Data-Science-B\14_module_machine_learning_classification"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Verify installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

#### macOS (Terminal)
```bash
# 1. Navigate to the module directory
cd "Data-Science-B/14_module_machine_learning_classification"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

### GPU Setup for RAPIDS

#### Check CUDA Installation
```bash
# Check CUDA version
nvcc --version

# Check GPU status
nvidia-smi

# Verify cuML installation (after activating environment)
python -c "import cuml; print(f'cuML version: {cuml.__version__}')"
python -c "print(f'CUDA available: {cuml.is_cuda_available()}')"
```

#### RAPIDS Installation Issues
If RAPIDS installation fails:
1. Ensure CUDA toolkit is properly installed
2. Check GPU compatibility (compute capability 6.0+)
3. Use CPU-only mode for algorithms without GPU acceleration
4. Follow RAPIDS installation guide: https://docs.rapids.ai/install

### Running the Scripts

#### Python Scripts
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run lecture demo
python "lecture_demo.py"

# Run student lab
python "student_lab.py"
```

#### Jupyter Notebooks
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run Jupyter notebook server
jupyter notebook

# Or run specific notebooks
jupyter notebook "lecture_demo.ipynb"
jupyter notebook "student_lab.ipynb"
```

### Dependencies Included

This environment includes:
- **scikit-learn 1.3.0+**: Core machine learning algorithms
- **XGBoost 1.7.0+**: Gradient boosting framework
- **RAPIDS cuML**: GPU-accelerated ML (if GPU available)
- **pandas 2.0.0+**: Data manipulation
- **numpy 1.24.0+**: Numerical computing
- **matplotlib 3.6.0+**: Data visualization
- **seaborn 0.12.0+**: Statistical visualization
- **jupyter 1.0.0+**: Interactive notebooks
- **ipykernel 6.0.0+**: Jupyter kernel

### GPU Memory Management

#### Monitor GPU Usage
```python
# In Python, monitor GPU memory
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU Memory: {info.used/1024**3:.1f}GB used / {info.total/1024**3:.1f}GB total")
```

#### Memory Optimization Tips
- Process data in batches for large datasets
- Use float32 instead of float64 when possible
- Clear GPU memory between operations
- Monitor memory usage during training

### Troubleshooting

#### Environment Issues
```bash
# Re-sync environment
uv sync --reinstall

# Clear cache and re-sync
uv cache clean
uv sync
```

#### GPU-Related Issues
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
nvcc --version

# Test cuML import
python -c "import cuml; print('cuML imported successfully')"
```

#### Memory Issues
- **Out of GPU memory**: Reduce batch sizes, use CPU fallback
- **Slow data transfer**: Minimize CPU-GPU transfers
- **Training instability**: Check data preprocessing and normalization

#### Deactivate Environment
```bash
# When done working
deactivate
```

### Dataset Requirements

The scripts expect sample datasets. Download if missing:
```bash
# Example datasets (modify as needed)
curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
curl -o iris.csv https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv
```

### Support

If you encounter issues:
1. Ensure you're in the correct directory
2. Verify the virtual environment is activated
3. Check that all dependencies are installed with `uv sync`
4. For GPU issues, verify CUDA installation and GPU compatibility
5. Monitor memory usage when working with large datasets

### Key ML Concepts Covered

- **Supervised Learning**: Training, validation, test sets
- **Classification Algorithms**: Linear models, trees, ensembles
- **Model Evaluation**: Confusion matrix, ROC-AUC, cross-validation
- **GPU Acceleration**: RAPIDS cuML for performance
- **Hyperparameter Tuning**: Grid search, cross-validation
- **Scalable ML**: Integration with Dask and distributed computing

### Performance Expectations

#### CPU-Only Setup
- Suitable for learning and small datasets
- All algorithms will work but may be slower
- Good for understanding concepts

#### GPU-Accelerated Setup
- Significant speedups for large datasets
- Required for RAPIDS-specific content
- Optimal for production-like performance

### Next Steps

After setup completion:
1. Run the lecture demo to understand key concepts
2. Complete the student lab exercises
3. Experiment with different algorithms and parameters
4. Explore GPU acceleration benefits on your datasets

---

**Instructor**: Dennis Omboga Mongare
**Last Updated**: October 2025
**Version**: 1.0