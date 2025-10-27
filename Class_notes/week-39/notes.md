# Module 14: Machine Learning Classification

## Instructor Information
**Lead Instructor**: Dennis Omboga Mongare
**Role**: Lead Data Science Instructor
**Course**: Data Science B 

## Module Overview

This module builds on the scalable computing foundations from Modules 11-13 (Spark, HBase, Dask) to introduce machine learning classification algorithms. Students will learn how to implement and scale classification models using distributed computing frameworks, focusing on decision trees, random forests, and ensemble methods with RAPIDS acceleration.

## Learning Objectives

- Understand supervised learning classification concepts
- Master decision tree algorithms and their implementation
- Implement ensemble methods: bagging, boosting, and random forests
- Apply cross-validation and model evaluation techniques
- Scale ML workflows using distributed computing (Spark, Dask)
- Optimize performance with RAPIDS GPU acceleration
- Evaluate model performance using ROC-AUC and confusion matrices

## Prerequisites

- Completion of Modules 11-13 (Spark, HBase, Dask)
- Understanding of distributed computing concepts
- Basic knowledge of Python data science libraries (pandas, numpy, scikit-learn)
- Familiarity with machine learning fundamentals

## Key Concepts

### 1. Supervised Learning Fundamentals
- Classification vs regression
- Training, validation, and test sets
- Overfitting and underfitting
- Bias-variance tradeoff

### 2. Decision Trees
- Tree construction algorithms (ID3, C4.5, CART)
- Splitting criteria (Gini impurity, entropy, information gain)
- Pruning techniques
- Interpretability advantages

### 3. Ensemble Methods
- **Bagging**: Bootstrap aggregation, random forests
- **Boosting**: AdaBoost, Gradient Boosting, XGBoost
- **Stacking**: Combining multiple models
- Variance reduction and bias reduction

### 4. Model Evaluation
- Confusion matrix components
- Precision, recall, F1-score
- ROC curves and AUC
- Cross-validation techniques

### 5. Scalable ML with Distributed Computing
- Distributed model training with Spark MLlib
- Parallel processing with Dask ML
- GPU acceleration with RAPIDS cuML
- Handling large datasets for ML

## Practical Applications

- Fraud detection systems
- Customer churn prediction
- Medical diagnosis classification
- Spam email filtering
- Credit risk assessment
- Image classification pipelines

## Required Software

- Python 3.13+
- scikit-learn 1.3.0+
- RAPIDS cuML (for GPU acceleration)
- PySpark MLlib (for distributed ML)
- Dask ML (for parallel ML)
- matplotlib, seaborn (visualization)
- UV package manager

## Lab Exercises

### Lab 1: Decision Tree Classification
- Implement decision trees from scratch
- Compare with scikit-learn implementation
- Visualize tree structures
- Hyperparameter tuning

### Lab 2: Random Forest Implementation
- Build random forest classifiers
- Feature importance analysis
- Out-of-bag error estimation
- Parallel training with Dask

### Lab 3: RAPIDS Accelerated ML
- GPU-accelerated decision trees
- RAPIDS cuML random forests
- Performance comparison with CPU implementations
- Memory optimization techniques

### Lab 4: Model Evaluation and Validation
- Cross-validation strategies
- ROC-AUC analysis
- Confusion matrix interpretation
- Model comparison and selection

### Lab 5: Distributed ML Pipelines
- Spark MLlib classification pipelines
- Dask ML parallel training
- Large-scale model deployment
- Performance monitoring

## Assessment Criteria

- Successful implementation of classification algorithms
- Understanding of ensemble method principles
- Effective use of distributed computing for ML
- Proper model evaluation and validation
- Performance optimization with RAPIDS
- Code quality and documentation

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [RAPIDS Documentation](https://docs.rapids.ai/)
- [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Dask ML Documentation](https://ml.dask.org/)

## Support

For technical issues or questions:
1. Check the setup.md file for environment configuration
2. Review scikit-learn and RAPIDS documentation
3. Consult with instructor during lab sessions
4. Use community forums for advanced topics

## Performance Optimization

- Memory management for large datasets
- GPU utilization with RAPIDS
- Distributed training strategies
- Hyperparameter optimization at scale

## Integration with Previous Modules

This module builds directly on:
- **Module 11 (Spark)**: Distributed data processing for ML pipelines
- **Module 12 (HBase)**: Scalable data storage for ML features
- **Module 13 (Dask)**: Parallel computing for ML model training

## Next Steps

After completing this module, students should be prepared for:
- Advanced machine learning techniques
- Deep learning with distributed computing
- MLOps and model deployment at scale
- Real-world ML system design

---

**Instructor**: Dennis Omboga Mongare
**Last Updated**: October 2025
**Version**: 1.0
