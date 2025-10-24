#!/usr/bin/env python3
"""
Student Spark Lab: Titanic Survival Analysis
Module 11: Scalable Computing - Apache Spark Ecosystem

This script provides a hands-on lab for students to learn Apache Spark
through practical exercises with the Titanic dataset.

Usage:
    python student_spark_lab.py

Prerequisites:
    pip install pyspark pandas numpy matplotlib seaborn

Terminal commands to run before starting:
    # 1. Check Python version
    python --version

    # 2. Install required packages
    pip install pyspark pandas numpy matplotlib seaborn

    # 3. Download Titanic dataset
    curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

    # 4. Verify Java installation (required for PySpark)
    java -version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import List, Tuple, Dict, Any

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def check_environment():
    """Check if all required packages are available"""
    print_header("Environment Check")

    # Check Python version
    print(f"Python version: {__import__('sys').version}")

    # Test basic imports
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("SUCCESS: Core data science libraries imported successfully")
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        print("Run: pip install pandas numpy matplotlib seaborn")
        return False

    # Test PySpark
    try:
        from pyspark.sql import SparkSession
        print("SUCCESS: PySpark imported successfully")
    except ImportError as e:
        print(f"ERROR: PySpark import error: {e}")
        print("Run: pip install pyspark")
        return False

    print("\nOBJECTIVE: Ready to start the Spark lab!")
    return True

def load_titanic_data() -> pd.DataFrame:
    """Load Titanic dataset with fallback to generated data"""
    try:
        df = pd.read_csv('titanic.csv')
        print(f"SUCCESS: Titanic dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(".2f")
        return df
    except FileNotFoundError:
        print("ERROR: titanic.csv not found. Download it first:")
        print("curl -o titanic.csv https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        print("📝 Using generated sample data for demo")

        # Create sample data for demo
        np.random.seed(42)
        df = pd.DataFrame({
            'PassengerId': range(1, 892),
            'Survived': np.random.choice([0, 1], 891),
            'Pclass': np.random.choice([1, 2, 3], 891),
            'Name': [f'Passenger_{i}' for i in range(891)],
            'Sex': np.random.choice(['male', 'female'], 891),
            'Age': np.random.normal(30, 15, 891).clip(0, 80),
            'Fare': np.random.exponential(30, 891)
        })
        return df

def exercise_1_spark_session(df: pd.DataFrame):
    """Exercise 1: Create and configure Spark Session"""
    print_header("Exercise 1: Spark Session Setup")

    print("OBJECTIVE: Task: Create a Spark session and load the Titanic data")
    print("💡 Hint: Use SparkSession.builder() with appropriate configurations")

    # TODO: Create Spark session
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("TitanicSparkLab") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .master("local[*]") \
        .getOrCreate()

    print(f"SUCCESS: Spark {spark.version} session created")

    # TODO: Load data into Spark DataFrame
    spark_df = spark.createDataFrame(df)
    print(f"SUCCESS: Spark DataFrame created: {spark_df.count()} rows")

    # TODO: Show schema
    print("\n📋 Data Schema:")
    spark_df.printSchema()

    # TODO: Show first 5 rows
    print("\nANALYSIS: First 5 rows:")
    spark_df.show(5)

    return spark, spark_df

def exercise_2_basic_operations(spark_df):
    """Exercise 2: Basic DataFrame Operations"""
    print_header("Exercise 2: Basic DataFrame Operations")

    print("OBJECTIVE: Tasks:")
    print("   1. Count total passengers")
    print("   2. Filter passengers by class")
    print("   3. Select specific columns")
    print("   4. Add a new calculated column")

    # TODO: Count total passengers
    total_passengers = spark_df.count()
    print(f"ANALYSIS: Total passengers: {total_passengers}")

    # TODO: Filter first-class passengers
    first_class = spark_df.filter(spark_df.Pclass == 1)
    print(f"🎫 First-class passengers: {first_class.count()}")

    # TODO: Select name, age, and fare columns
    selected_cols = spark_df.select("Name", "Age", "Fare")
    print("📋 Selected columns (first 3 rows):")
    selected_cols.show(3)

    # TODO: Add fare category column
    from pyspark.sql.functions import when, col

    categorized_df = spark_df.withColumn(
        "FareCategory",
        when(col("Fare") < 10, "Budget")
        .when(col("Fare") < 50, "Standard")
        .otherwise("Premium")
    )

    print("💰 Fare categories (first 5 rows):")
    categorized_df.select("Name", "Fare", "FareCategory").show(5)

    return categorized_df

def exercise_3_aggregation(spark_df):
    """Exercise 3: Aggregation Operations"""
    print_header("Exercise 3: Aggregation Operations")

    print("OBJECTIVE: Tasks:")
    print("   1. Calculate survival rate by passenger class")
    print("   2. Find average age by gender")
    print("   3. Count passengers by embarkation port")
    print("   4. Calculate statistics for fare by class")

    # TODO: Survival rate by class
    from pyspark.sql.functions import avg, count, mean

    survival_by_class = spark_df.groupBy("Pclass").agg(
        mean("Survived").alias("survival_rate"),
        count("*").alias("total_passengers")
    ).orderBy("Pclass")

    print("🏊 Survival rate by class:")
    survival_by_class.show()

    # TODO: Average age by gender
    age_by_gender = spark_df.groupBy("Sex").agg(
        mean("Age").alias("avg_age"),
        count("*").alias("count")
    )

    print("🎂 Average age by gender:")
    age_by_gender.show()

    # TODO: Count by embarkation port (if column exists)
    if "Embarked" in spark_df.columns:
        embarked_count = spark_df.groupBy("Embarked").count().orderBy("count", ascending=False)
        print("🚢 Passengers by embarkation port:")
        embarked_count.show()

    # TODO: Fare statistics by class
    fare_stats = spark_df.groupBy("Pclass").agg(
        mean("Fare").alias("avg_fare"),
        count("*").alias("count")
    ).orderBy("Pclass")

    print("💵 Average fare by class:")
    fare_stats.show()

    return survival_by_class

def exercise_4_spark_sql(spark_df):
    """Exercise 4: Spark SQL Queries"""
    print_header("Exercise 4: Spark SQL Queries")

    print("OBJECTIVE: Tasks:")
    print("   1. Create a temporary view")
    print("   2. Write SQL queries for complex analysis")
    print("   3. Join operations (if applicable)")

    # TODO: Create temporary view
    spark_df.createOrReplaceTempView("titanic")
    print("SUCCESS: Temporary view 'titanic' created")

    # TODO: SQL query for survival analysis
    sql_query1 = """
    SELECT
        Pclass,
        Sex,
        COUNT(*) as total_passengers,
        AVG(Survived) as survival_rate,
        AVG(Age) as avg_age
    FROM titanic
    GROUP BY Pclass, Sex
    ORDER BY Pclass, Sex
    """

    result1 = spark.sql(sql_query1)
    print("ANALYSIS: SQL Query 1 - Survival by class and gender:")
    result1.show()

    # TODO: SQL query for fare analysis
    sql_query2 = """
    SELECT
        Pclass,
        MIN(Fare) as min_fare,
        MAX(Fare) as max_fare,
        AVG(Fare) as avg_fare,
        COUNT(*) as passenger_count
    FROM titanic
    WHERE Fare > 0
    GROUP BY Pclass
    ORDER BY Pclass
    """

    result2 = spark.sql(sql_query2)
    print("💰 SQL Query 2 - Fare statistics by class:")
    result2.show()

    return result1, result2

def exercise_5_rdd_operations(spark_df):
    """Exercise 5: RDD Operations"""
    print_header("Exercise 5: RDD Operations")

    print("OBJECTIVE: Tasks:")
    print("   1. Convert DataFrame to RDD")
    print("   2. Perform map operations")
    print("   3. Perform filter operations")
    print("   4. Perform reduce operations")

    # TODO: Convert to RDD
    rdd = spark_df.rdd
    print(f"SUCCESS: RDD created with {rdd.getNumPartitions()} partitions")

    # TODO: Map operation - extract passenger classes
    classes_rdd = rdd.map(lambda row: row.Pclass)
    unique_classes = classes_rdd.distinct().collect()
    print(f"🎫 Unique passenger classes: {sorted(unique_classes)}")

    # TODO: Filter operation - adult passengers
    adults_rdd = rdd.filter(lambda row: row.Age and row.Age >= 18)
    adults_count = adults_rdd.count()
    print(f"👨‍ Adult passengers (18+): {adults_count}")

    # TODO: Map-Reduce operation - total fare by class
    fare_by_class = rdd \
        .map(lambda row: (row.Pclass, row.Fare or 0)) \
        .reduceByKey(lambda x, y: x + y) \
        .collect()

    print("💵 Total fare collected by class:")
    for pclass, total_fare in sorted(fare_by_class):
        print(".2f")

    return fare_by_class

def exercise_6_visualization(df: pd.DataFrame, spark_results):
    """Exercise 6: Data Visualization"""
    print_header("Exercise 6: Data Visualization")

    print("OBJECTIVE: Tasks:")
    print("   1. Create survival rate bar chart")
    print("   2. Create age distribution histogram")
    print("   3. Create scatter plot of age vs fare")
    print("   4. Save visualizations")

    # TODO: Survival rate by class
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    survival_by_class = df.groupby('Pclass')['Survived'].mean()
    axes[0, 0].bar(survival_by_class.index, survival_by_class.values,
                   color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
    axes[0, 0].set_title('Survival Rate by Class')
    axes[0, 0].set_ylabel('Survival Rate')
    axes[0, 0].set_xticks([1, 2, 3])

    # TODO: Age distribution
    df['Age'].hist(bins=30, ax=axes[0, 1], alpha=0.7, color='skyblue')
    axes[0, 1].set_title('Age Distribution')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Frequency')

    # TODO: Age vs Fare scatter plot
    axes[1, 0].scatter(df['Age'], df['Fare'], alpha=0.5, color='green')
    axes[1, 0].set_title('Age vs Fare')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Fare')

    # TODO: Gender distribution
    gender_counts = df['Sex'].value_counts()
    axes[1, 1].pie(gender_counts.values, labels=gender_counts.index,
                   autopct='%1.1f%%', colors=['lightblue', 'pink'])
    axes[1, 1].set_title('Gender Distribution')

    plt.tight_layout()
    plt.savefig('student_spark_lab_results.png', dpi=150, bbox_inches='tight')
    print("ANALYSIS: Visualizations saved as 'student_spark_lab_results.png'")
    plt.show()

def main():
    """Main lab function"""
    print("🚀 Student Spark Lab: Titanic Survival Analysis")
    print("=" * 55)
    print("Module 11: Scalable Computing - Apache Spark Ecosystem")
    print("=" * 55)

    # Check environment
    if not check_environment():
        print("ERROR: Environment check failed. Please install required packages.")
        return

    # Load data
    df = load_titanic_data()

    # Exercise 1: Spark Session
    spark, spark_df = exercise_1_spark_session(df)

    # Exercise 2: Basic Operations
    categorized_df = exercise_2_basic_operations(spark_df)

    # Exercise 3: Aggregation
    survival_results = exercise_3_aggregation(spark_df)

    # Exercise 4: Spark SQL
    sql_results = exercise_4_spark_sql(spark_df)

    # Exercise 5: RDD Operations
    rdd_results = exercise_5_rdd_operations(spark_df)

    # Exercise 6: Visualization
    exercise_6_visualization(df, survival_results)

    # Cleanup
    spark.stop()

    # Summary
    print_header("Lab Summary")

    print("OBJECTIVE: What You Learned:")
    print("1. SUCCESS: Spark Session creation and configuration")
    print("2. SUCCESS: DataFrame basic operations (filter, select, withColumn)")
    print("3. SUCCESS: Aggregation operations (groupBy, agg)")
    print("4. SUCCESS: Spark SQL queries")
    print("5. SUCCESS: RDD operations (map, filter, reduce)")
    print("6. SUCCESS: Data visualization with matplotlib")

    print("\n🏁 Spark Lab completed successfully!")
    print("💡 Challenge: Try modifying the exercises with different datasets!")

if __name__ == "__main__":
    main()