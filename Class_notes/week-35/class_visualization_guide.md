# Class Experiment: Running Hadoop/MapReduce Demos

## Overview
This guide helps instructors run interactive Hadoop/MapReduce demonstrations during class, showing students how to execute code cell-by-cell and visualize the concepts in real-time.

## Prerequisites
- Jupyter Lab installed (`pip install jupyterlab`)
- All required packages: `pyspark`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- Titanic dataset (`titanic.csv`) in the same directory

## Quick Start Commands

### Launch Jupyter Lab
```bash
# Navigate to the instructions directory
cd /path/to/Data-Science-B/10_Module-Scalable-Computing-Hadoop-Hive/Labs/Module-10-Lab-hadoop/instructions/

# Launch Jupyter Lab
jupyter lab
```

### Alternative: Run in VS Code
1. Open VS Code
2. Navigate to the instructions folder
3. Right-click on `student_hadoop_demo.ipynb` or `hadoop_lecture_demo.ipynb`
4. Select "Open with Jupyter"

## Class Experiment Structure

### Phase 1: Environment Setup (5 minutes)
**Objective:** Ensure all students can run the notebooks

1. **Open the notebook** in Jupyter Lab
2. **Run the first cell** (imports and setup)
3. **Check environment** - all packages should load successfully
4. **Discuss any errors** - help students troubleshoot

**Teaching Points:**
- Package management in data science
- Environment consistency
- Error handling importance

### Phase 2: Data Loading (3 minutes)
**Objective:** Show data ingestion and basic exploration

1. **Execute data loading cell**
2. **Show dataset statistics** - row count, columns, memory usage
3. **Discuss data quality** - missing values, data types

**Teaching Points:**
- Data ingestion patterns
- Memory considerations
- Dataset validation

### Phase 3: Single Machine Limitations (5 minutes)
**Objective:** Demonstrate why distributed computing is needed

1. **Run basic analysis** - survival rates, class distributions
2. **Show "big data" simulation** - 1 billion passenger calculation
3. **Discuss scaling problems** - memory, processing time

**Teaching Points:**
- Big data challenges
- Single machine limitations
- Need for distributed systems

### Phase 4: Manual MapReduce (10 minutes)
**Objective:** Implement MapReduce from scratch

1. **MAP Phase:** Extract key-value pairs
   - Show mapper function
   - Execute mapping step
   - Display intermediate results

2. **SHUFFLE Phase:** Group by keys
   - Show shuffle logic
   - Execute grouping
   - Visualize data distribution

3. **REDUCE Phase:** Aggregate results
   - Show reducer function
   - Calculate survival rates
   - Display final results

**Teaching Points:**
- MapReduce paradigm
- Parallel processing concepts
- Data transformation patterns

### Phase 5: PySpark Comparison (8 minutes)
**Objective:** Show modern distributed processing

1. **Initialize Spark Session**
2. **Load data into Spark DataFrame**
3. **Execute SQL queries** (equivalent to HiveQL)
4. **Compare with manual results**

**Teaching Points:**
- High-level APIs vs. low-level implementation
- Spark ecosystem
- SQL in big data

### Phase 6: Performance Analysis (5 minutes)
**Objective:** Compare different approaches

1. **Run pandas performance test**
2. **Show Spark performance** (if available)
3. **Discuss trade-offs**

**Teaching Points:**
- Tool selection criteria
- Performance characteristics
- Scaling considerations

### Phase 7: Visualizations (5 minutes)
**Objective:** Create and interpret charts

1. **Generate survival rate plots**
2. **Show demographic analysis**
3. **Save results**

**Teaching Points:**
- Data visualization in distributed systems
- Result interpretation
- Communication of insights

### Phase 8: Java Connection (5 minutes)
**Objective:** Link to students' Java lab

1. **Show Java MapReduce equivalent**
2. **Compare Python vs Java approaches**
3. **Discuss implementation differences**

**Teaching Points:**
- Cross-language concepts
- Hadoop ecosystem
- Industry applications

## Interactive Teaching Tips

### Cell-by-Cell Execution
- **Pause after each cell** to explain what's happening
- **Show intermediate results** to build understanding
- **Encourage questions** at natural breakpoints

### Error Handling
- **Expect PySpark issues** in some environments
- **Have fallback explanations** ready
- **Use errors as teaching moments**

### Student Engagement
- **Ask prediction questions:** "What do you think this cell will output?"
- **Encourage modifications:** "Try changing this parameter"
- **Relate to real world:** "Where would you use this in industry?"

## Troubleshooting Common Issues

### Jupyter Won't Start
```bash
# Kill existing processes
pkill -f jupyter

# Clear cache
rm -rf ~/.jupyter/runtime/

# Restart
jupyter lab --no-browser
```

### PySpark Not Working
- **Expected in many environments**
- **Use as teaching opportunity:** "This is why we have fallback strategies"
- **Focus on concepts rather than execution**

### Memory Issues
- **Close other applications**
- **Restart kernel if needed**
- **Use smaller datasets for demos**

### Package Import Errors
```bash
# Update packages
pip install --upgrade pyspark pandas numpy matplotlib seaborn

# Or use conda
conda update --all
```

## Alternative Demonstration Methods

### Method 1: Live Coding (Recommended)
- **Instructor types code live** while explaining
- **Students follow along** in their own notebooks
- **Interactive Q&A** throughout

### Method 2: Pre-executed Demo
- **Run notebook beforehand** with all outputs saved
- **Walk through results** cell by cell
- **Focus on explanation** rather than execution

### Method 3: Student Hands-on
- **Provide notebook skeleton**
- **Students fill in code** with guidance
- **Peer learning** and collaboration

## Timing Guidelines

| Phase | Time | Cumulative |
|-------|------|------------|
| Setup | 5 min | 5 min |
| Data Loading | 3 min | 8 min |
| Single Machine Limits | 5 min | 13 min |
| Manual MapReduce | 10 min | 23 min |
| PySpark Demo | 8 min | 31 min |
| Performance | 5 min | 36 min |
| Visualizations | 5 min | 41 min |
| Java Connection | 5 min | 46 min |

**Total: ~45 minutes** (adjust based on class level and questions)

## Key Learning Outcomes

By the end of this demonstration, students should understand:

1. **Big Data Problems:** Why single machines can't handle massive datasets
2. **MapReduce Paradigm:** map → shuffle → reduce pattern
3. **Distributed Concepts:** Parallel processing and fault tolerance
4. **Tool Comparison:** pandas vs PySpark vs Java MapReduce
5. **Real Applications:** How these concepts work in industry

## Materials Needed

- **Projector/Whiteboard** for displaying notebook
- **Student devices** with Jupyter installed (optional)
- **Internet connection** for package downloads
- **Backup plan** if PySpark doesn't work

## Success Metrics

- **Students can explain** the three MapReduce phases
- **Students understand** when to use different tools
- **Students can relate** concepts to their Java lab
- **Students see** the value of distributed computing

Happy teaching!