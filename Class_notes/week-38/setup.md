# Module 13: Scalable Computing with Dask and UCX - Setup Guide

## Instructor Information
**Lead Instructor**: Dennis Omboga Mongare
**Role**: Lead Data Science Instructor
**Course**: Data Science B - Scalable Computing Series

## Environment Setup with UV

This module uses `uv` for dependency management and virtual environment isolation. Follow these steps to set up your environment.

### Prerequisites
- Python 3.13 or higher
- `uv` package manager (install with: `pip install uv` or follow [uv installation guide](https://github.com/astral-sh/uv))

### Setup Commands

#### Linux (Cinnamon Manjaro)
```bash
# 1. Navigate to the module directory
cd "Data-Science-B/13_Module_Scalable-Computing-Dask-and-UCX"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import dask; print(f'Dask version: {dask.__version__}')"
python -c "import dask.dataframe as dd; import dask.array as da; print('Dask components imported successfully')"
python -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('Core libraries imported successfully')"
```

#### Windows (PowerShell/Command Prompt)
```powershell
# 1. Navigate to the module directory
cd "Data-Science-B\13_Module_Scalable-Computing-Dask-and-UCX"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Verify installation
python -c "import dask; print(f'Dask version: {dask.__version__}')"
python -c "import dask.dataframe as dd; import dask.array as da; print('Dask components imported successfully')"
python -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('Core libraries imported successfully')"
```

#### macOS (Terminal)
```bash
# 1. Navigate to the module directory
cd "Data-Science-B/13_Module_Scalable-Computing-Dask-and-UCX"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import dask; print(f'Dask version: {dask.__version__}')"
python -c "import dask.dataframe as dd; import dask.array as da; print('Dask components imported successfully')"
python -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('Core libraries imported successfully')"
```

### Running the Scripts

#### Python Scripts
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run Dask lecture demo
python "Quiz Questions and Answers/dask_lecture_demo.py"

# Run Dask student lab
python "Quiz Questions and Answers/student_dask_lab.py"
```

#### Jupyter Notebooks
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run Jupyter notebook server
jupyter notebook

# Or run specific notebooks
jupyter notebook "Quiz Questions and Answers/dask_lecture_demo.ipynb"
jupyter notebook "Quiz Questions and Answers/student_dask_lab.ipynb"
```

### Dependencies Included

This environment includes:
- **Dask 2024.1.0+** with extras:
  - `dask[dataframe]`: DataFrame operations
  - `dask[array]`: Array operations
  - `dask[distributed]`: Distributed computing
- **pandas 2.0.0+**: Data manipulation
- **numpy 1.24.0+**: Numerical computing
- **matplotlib 3.6.0+**: Data visualization
- **seaborn 0.12.0+**: Statistical visualization
- **scikit-learn 1.3.0+**: Machine learning
- **jupyter 1.0.0+**: Interactive notebooks
- **ipykernel 6.0.0+**: Jupyter kernel

### Dask Cluster Setup (Optional)

For distributed computing demonstrations:

#### Local Cluster (Recommended for learning)
```bash
# Start a local Dask cluster (from within activated environment)
python -c "
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=2, threads_per_worker=1)
client = Client(cluster)
print(f'Dask dashboard available at: {client.dashboard_link}')
"
```

#### Distributed Cluster (Advanced)
```bash
# Install additional distributed dependencies if needed
uv add dask[distributed]

# Start scheduler
dask-scheduler

# Start workers (in separate terminals)
dask-worker localhost:8786
dask-worker localhost:8786
```

### Troubleshooting

#### Environment Issues
```bash
# Re-sync environment
uv sync --reinstall

# Clear cache and re-sync
uv cache clean
uv sync
```

#### Memory Issues with Large Datasets
Dask handles large datasets, but monitor memory usage:
```bash
# Check system memory
free -h

# Monitor Dask memory usage
python -c "import psutil; print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')"
```

#### Performance Optimization
```bash
# Set Dask configuration for better performance
export DASK_DATAFRAME__QUERY_PLANNING=True
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=30s
```

#### Deactivate Environment
```bash
# When done working
deactivate
```

### Dataset Requirements

The scripts expect a `titanic.csv` file in the working directory. Download it if missing:
```bash
curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

### Support

If you encounter issues:
1. Ensure you're in the correct directory
2. Verify the virtual environment is activated
3. Check that all dependencies are installed with `uv sync`
4. For distributed computing, start with local cluster examples
5. Monitor memory usage when working with large datasets

### Key Dask Concepts Covered

- **Dask Arrays**: Parallel numpy-like operations
- **Dask DataFrames**: Parallel pandas-like operations
- **Dask Bags**: Parallel processing of Python objects
- **Distributed Computing**: Scaling to multiple machines
- **Lazy Evaluation**: Building computation graphs
- **Task Scheduling**: Optimizing parallel execution