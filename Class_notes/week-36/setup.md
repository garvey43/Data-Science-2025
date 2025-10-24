# Module 11: Scalable Computing with Spark - Setup Guide

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
cd "Data-Science-B/11_Module-Scalable-Computing-Spark"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

#### Windows (PowerShell/Command Prompt)
```powershell
# 1. Navigate to the module directory
cd "Data-Science-B\11_Module-Scalable-Computing-Spark"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Verify installation
python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

#### macOS (Terminal)
```bash
# 1. Navigate to the module directory
cd "Data-Science-B/11_Module-Scalable-Computing-Spark"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

### Running the Scripts

#### Python Scripts
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run Spark lecture demo
python "Labs/spark_lecture_demo.py"

# Run Spark student lab
python "Labs/student_spark_lab.py"
```

#### Jupyter Notebooks
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run Jupyter notebook server
jupyter notebook

# Or run specific notebooks
jupyter notebook "Labs/spark_lecture_demo.ipynb"
jupyter notebook "Labs/student_spark_lab.ipynb"
```

### Dependencies Included

This environment includes:
- **PySpark 3.5.0+**: Apache Spark Python API
- **pandas 2.0.0+**: Data manipulation
- **numpy 1.24.0+**: Numerical computing
- **matplotlib 3.6.0+**: Data visualization
- **seaborn 0.12.0+**: Statistical visualization
- **jupyter 1.0.0+**: Interactive notebooks
- **ipykernel 6.0.0+**: Jupyter kernel

### Troubleshooting

#### Java Requirement for Spark

##### Linux (Cinnamon Manjaro)
Spark requires Java. Install OpenJDK if not present:
```bash
# Check Java
java -version

# Install Java (if needed)
sudo pacman -S jdk-openjdk  # On Manjaro/Arch
```

##### Windows
```powershell
# Check Java
java -version

# Download and install from: https://adoptium.net/
# Or use Chocolatey: choco install openjdk
```

##### macOS
```bash
# Check Java
java -version

# Install using Homebrew
brew install openjdk

# Or download from: https://adoptium.net/
```

#### Environment Issues
```bash
# Re-sync environment
uv sync --reinstall

# Clear cache and re-sync
uv cache clean
uv sync
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
4. Confirm Java is installed for Spark functionality