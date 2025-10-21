# Student Run Instructions for Hadoop & MapReduce Demo

## Overview
This demo introduces you to MapReduce concepts through hands-on Python implementation, bridging the gap between your Java Hadoop lab and practical data science workflows.

## Prerequisites
Before running the demo, ensure you have:

1. **Python 3.7+** installed
2. **Required packages**: `pyspark`, `pandas`, `numpy`, `matplotlib`, `seaborn`
3. **Titanic dataset**: `titanic.csv` in the same directory
4. **Java JDK 11+** (for PySpark functionality)

## Quick Setup (One Command)

### Option 1: Using pip (Recommended for most students)
```bash
# Install all required packages
pip install pyspark pandas numpy matplotlib seaborn

# Download the dataset
curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# Run the demo
python student_hadoop_demo.py
```

### Option 2: Using conda (If you prefer conda environments)
```bash
# Create and activate environment
conda create -n hadoop-demo python=3.9
conda activate hadoop-demo

# Install packages
conda install -c conda-forge pyspark pandas numpy matplotlib seaborn

# Download dataset and run
curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
python student_hadoop_demo.py
```

### Option 3: Google Colab (Easiest - No setup required)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `student_hadoop_demo.py`
3. Run all cells - it will handle setup automatically

## Step-by-Step Instructions

### Step 1: Environment Check
The script automatically checks your environment when you run it. Look for:
- ✅ Python version confirmation
- ✅ Package import success messages
- ✅ PySpark availability

### Step 2: Data Loading
The demo loads the Titanic dataset. If `titanic.csv` is missing, it will:
- Show download instructions
- Use generated sample data as fallback

### Step 3: Demo Flow
The demo runs through 6 parts automatically:

1. **Single Machine Limitations** - Why we need distributed computing
2. **Manual MapReduce** - Implementing map/shuffle/reduce from scratch
3. **PySpark Implementation** - Modern distributed processing
4. **Performance Analysis** - Comparing approaches
5. **Visual Analysis** - Charts and graphs
6. **Java Connection** - Linking to your Hadoop lab

### Step 4: Output Files
After running, you'll get:
- `student_hadoop_demo_results.png` - Visualization charts
- Console output with detailed explanations

## Troubleshooting

### Common Issues

**PySpark not working?**
- Install Java JDK 11+ from [oracle.com](https://www.oracle.com/java/)
- Set JAVA_HOME environment variable
- The demo will still work with pandas-only mode

**Missing titanic.csv?**
```bash
curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

**Import errors?**
```bash
pip install --upgrade pyspark pandas numpy matplotlib seaborn
```

**Permission errors on Windows?**
- Run terminal/command prompt as Administrator
- Or use `pip install --user` for user-level installation

### Environment-Specific Help

**Windows:**
```bash
# Install Python from python.org
# Use PowerShell or Command Prompt
pip install pyspark pandas numpy matplotlib seaborn
```

**macOS:**
```bash
# Install Homebrew first: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python openjdk@11
pip install pyspark pandas numpy matplotlib seaborn
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip default-jdk
pip install pyspark pandas numpy matplotlib seaborn
```

## Expected Runtime
- **Full demo**: 2-5 minutes
- **Without PySpark**: 1-2 minutes
- **With large datasets**: 5-10+ minutes

## Learning Objectives
After running this demo, you should understand:
- Why distributed computing is necessary for big data
- The map → shuffle → reduce paradigm
- How PySpark simplifies MapReduce operations
- Performance trade-offs between different approaches
- Connection between Python concepts and Java MapReduce

## Next Steps
- Try modifying the mapper/reducer functions
- Experiment with different datasets
- Compare performance on larger data (1M+ rows)
- Explore PySpark DataFrame operations

## Support
If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Try running in Google Colab (no setup needed)
4. Ask your instructor for environment-specific help

Happy learning! 🚀
