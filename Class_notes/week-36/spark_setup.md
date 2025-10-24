# Setup Guide for Module 11: Scalable Computing - Apache Spark

## Overview
This guide will help you set up your development environment for the Apache Spark hands-on demo and lab. We'll cover multiple setup options to accommodate different operating systems and preferences.

## Prerequisites Checklist

### Required Software
- [ ] Python 3.7 or higher
- [ ] Java JDK 11 or higher (for PySpark)
- [ ] Git (optional, for cloning repositories)

### Required Python Packages
- [ ] pyspark
- [ ] pandas
- [ ] numpy
- [ ] matplotlib
- [ ] seaborn

### Required Data
- [ ] titanic.csv dataset

## Setup Options

### Option 1: UV Environment (Recommended - Already Configured)

If you're using the provided UV environment:

```bash
# The environment is already set up with all dependencies
# Just activate and run
uv run --python hadoop_lecture_env/bin/python python spark_lecture_demo.py
```

### Option 2: pip Installation (Most Common)

#### Windows Setup
```bash
# 1. Install Python from https://python.org (if not already installed)
# 2. Open Command Prompt or PowerShell as Administrator

# Install required packages
pip install pyspark pandas numpy matplotlib seaborn

# Download the dataset
curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# Verify installation
python -c "import pyspark, pandas, numpy, matplotlib, seaborn; print('All packages installed successfully!')"

# Run the demo
python spark_lecture_demo.py
```

#### macOS Setup
```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python and Java
brew install python openjdk@11

# 3. Install packages
pip install pyspark pandas numpy matplotlib seaborn

# 4. Download dataset
curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# 5. Run demo
python spark_lecture_demo.py
```

#### Linux Setup (Ubuntu/Debian)
```bash
# 1. Update package list
sudo apt update

# 2. Install Python and Java
sudo apt install python3 python3-pip default-jdk

# 3. Install packages
pip install pyspark pandas numpy matplotlib seaborn

# 4. Download dataset
curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# 5. Run demo
python spark_lecture_demo.py
```

### Option 3: Conda Environment (Alternative)

#### Create Conda Environment
```bash
# Create new environment
conda create -n spark-demo python=3.9

# Activate environment
conda activate spark-demo

# Install packages
conda install -c conda-forge pyspark pandas numpy matplotlib seaborn

# Download dataset
curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# Run demo
python spark_lecture_demo.py
```

### Option 4: Google Colab (No Setup Required)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. Upload `spark_lecture_demo.py` or `student_spark_lab.py`
4. The notebook will handle all setup automatically

## Environment Verification

Run this verification script to ensure everything is working:

```python
# =============================================
# SPARK ENVIRONMENT VERIFICATION SCRIPT
# =============================================

print("🔍 Verifying Spark Environment...")

# Check Python version
import sys
print(f"Python version: {sys.version}")

# Check core packages
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("SUCCESS: Core data science packages: OK")
except ImportError as e:
    print(f"ERROR: Core packages missing: {e}")

# Check PySpark
try:
    from pyspark.sql import SparkSession
    print("SUCCESS: PySpark: OK")

    # Test Spark session creation
    spark = SparkSession.builder \
        .appName("VerificationTest") \
        .master("local[*]") \
        .getOrCreate()

    print("SUCCESS: Spark session creation: OK")
    spark.stop()

except Exception as e:
    print(f"ERROR: PySpark issue: {e}")

# Check Java
import subprocess
try:
    result = subprocess.run(['java', '-version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("SUCCESS: Java: OK")
    else:
        print("ERROR: Java: Missing or not in PATH")
except:
    print("ERROR: Java: Not found")

# Check dataset
import os
if os.path.exists('titanic.csv'):
    print("SUCCESS: Dataset: OK")
else:
    print("ERROR: Dataset: titanic.csv not found")

print("\nOBJECTIVE: If all checks pass, you're ready for Spark!")
```

## Troubleshooting Common Issues

### PySpark Installation Issues

**Problem:** `pip install pyspark` fails
**Solution:**
```bash
# Try upgrading pip first
pip install --upgrade pip

# Install with specific version
pip install pyspark==3.5.0

# Or use conda
conda install -c conda-forge pyspark
```

### Java Issues

**Problem:** PySpark fails with Java errors
**Solutions:**
```bash
# Windows: Set JAVA_HOME
set JAVA_HOME="C:\Program Files\Java\jdk-11"

# macOS/Linux: Check Java version
java -version

# Install Java if missing
# Windows: Download from oracle.com
# macOS: brew install openjdk@11
# Linux: sudo apt install default-jdk
```

### Memory Issues

**Problem:** Spark runs out of memory
**Solutions:**
```bash
# Reduce driver memory
spark = SparkSession.builder \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

# Or use local mode with limited cores
.master("local[2]")
```

### Dataset Download Issues

**Problem:** curl fails to download titanic.csv
**Alternatives:**
```bash
# Use wget instead
wget https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# Or download manually from browser
# https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# Or use Python to download
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv', 'titanic.csv')"
```

### Permission Errors

**Problem:** Permission denied when installing packages
**Solutions:**
```bash
# Use --user flag
pip install --user pyspark pandas numpy matplotlib seaborn

# Or run as administrator/sudo
sudo pip install pyspark pandas numpy matplotlib seaborn
```

### Virtual Environment Issues

**Problem:** Packages installed in wrong environment
**Solution:**
```bash
# Check which Python you're using
which python
python -c "import sys; print(sys.executable)"

# Activate correct environment
# For conda: conda activate spark-demo
# For venv: source venv/bin/activate
```

## Performance Considerations

### Memory Requirements
- **Minimum:** 4GB RAM
- **Recommended:** 8GB+ RAM for PySpark
- **Large datasets:** 16GB+ RAM

### Storage Requirements
- **Demo:** ~50MB (packages + dataset)
- **Development:** ~2GB (full Spark setup)

## IDE Recommendations

### VS Code (Recommended)
1. Install Python extension
2. Install Jupyter extension
3. Open folder containing demo files
4. Run in integrated terminal

### PyCharm
1. Create new project
2. Configure interpreter
3. Install packages via settings
4. Run script in terminal

### Jupyter Lab
```bash
# Install Jupyter
pip install jupyterlab

# Launch
jupyter lab

# Open spark_lecture_demo.py as notebook
```

## Testing Your Setup

After setup, run the demo to verify everything works:

```bash
python spark_lecture_demo.py
```

**Expected output:**
- Environment check passes
- Dataset loads successfully
- Spark operations complete
- Performance comparisons shown
- Visualizations are generated

## Getting Help

If you encounter issues:

1. **Check this guide first** - Most common problems are covered
2. **Run the verification script** - Identifies specific issues
3. **Try Google Colab** - No setup required, guaranteed to work
4. **Check online resources:**
   - PySpark documentation: https://spark.apache.org/docs/latest/
   - Stack Overflow for specific errors
   - Your course forum for peer help

## Next Steps

Once your environment is set up:

1. Run the lecture demo and observe the output
2. Complete the student lab exercises
3. Experiment with different Spark operations
4. Explore Spark SQL and DataFrame APIs

Happy learning with Spark! 🚀