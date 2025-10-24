# Module 12: Scalable Computing with HBase - Setup Guide

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
cd "Data-Science-B/12_Module-Scalable-Computing-HBase"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import happybase; print('HappyBase imported successfully')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

#### Windows (PowerShell/Command Prompt)
```powershell
# 1. Navigate to the module directory
cd "Data-Science-B\12_Module-Scalable-Computing-HBase"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Verify installation
python -c "import happybase; print('HappyBase imported successfully')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

#### macOS (Terminal)
```bash
# 1. Navigate to the module directory
cd "Data-Science-B/12_Module-Scalable-Computing-HBase"

# 2. Sync dependencies and create virtual environment
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import happybase; print('HappyBase imported successfully')"
python -c "import pandas, numpy, matplotlib, seaborn; print('Core libraries imported successfully')"
```

### Running the Scripts

#### Python Scripts
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run HBase lecture demo
python "Quiz Questions and Answers/hbase_lecture_demo.py"

# Run HBase student lab
python "Quiz Questions and Answers/student_hbase_lab.py"
```

#### Jupyter Notebooks
```bash
# Activate environment (if not already activated)
source .venv/bin/activate

# Run Jupyter notebook server
jupyter notebook

# Or run specific notebooks
jupyter notebook "Quiz Questions and Answers/hbase_lecture_demo.ipynb"
jupyter notebook "Quiz Questions and Answers/student_hbase_lab.ipynb"
```

### Dependencies Included

This environment includes:
- **HappyBase 1.2.0+**: HBase Python client
- **pandas 2.0.0+**: Data manipulation
- **numpy 1.24.0+**: Numerical computing
- **matplotlib 3.6.0+**: Data visualization
- **seaborn 0.12.0+**: Statistical visualization
- **jupyter 1.0.0+**: Interactive notebooks
- **ipykernel 6.0.0+**: Jupyter kernel
- **requests 2.31.0+**: HTTP library for data downloads

### HBase Server Setup (Optional)

For full HBase functionality, you'll need a running HBase instance. For learning purposes, the scripts include fallback modes that work without HBase.

#### Quick HBase Setup with Docker (Recommended for learning)
```bash
# Install Docker
sudo pacman -S docker

# Start Docker service
sudo systemctl start docker

# Run HBase in Docker
docker run -d -p 9090:9090 --name hbase harisekhon/hbase

# Check if HBase is running
docker ps
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

#### HBase Connection Issues
If HBase connection fails (expected in many environments):
- The scripts will automatically fall back to conceptual exercises
- Focus on understanding HBase concepts rather than live connections
- Use the provided pandas-based alternatives for data operations

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
4. HBase connectivity is optional - the scripts work in demo mode without it