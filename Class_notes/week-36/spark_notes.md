# Module 11 Theory Notes: Scalable Computing - Apache Spark Ecosystem

## Overview
Module 11 builds on the MapReduce foundations from Module 10, introducing Apache Spark as the modern evolution of distributed data processing. Spark addresses Hadoop's limitations while maintaining compatibility with the broader big data ecosystem.

## Core Concepts

### 1. Spark vs Hadoop Evolution

**Hadoop MapReduce Limitations:**
- Disk-based processing (slow I/O operations)
- Rigid two-phase paradigm (Map → Reduce only)
- High latency for iterative algorithms
- Complex for interactive queries
- Verbose Java API

**Spark Innovations:**
- In-memory processing (100x faster than disk-based)
- DAG-based execution engine (flexible processing graphs)
- Rich ecosystem (SQL, Streaming, ML, Graph)
- Unified API across languages (Python, Scala, Java, R)
- Interactive shell and notebooks support

### 2. Spark Architecture

#### Core Components
- **Driver Program:** Main application that creates SparkContext
- **Cluster Manager:** Allocates resources (YARN, Kubernetes, Mesos, Standalone)
- **Worker Nodes:** Execute tasks and store data
- **Executors:** JVM processes that run tasks on worker nodes
- **Tasks:** Individual units of work executed on executors

#### Execution Model
```
Driver Program → Cluster Manager → Worker Nodes → Executors → Tasks
```

#### Data Structures
- **RDD (Resilient Distributed Dataset):** Immutable, partitioned collection
- **DataFrame:** Distributed collection of data organized into named columns
- **Dataset:** Type-safe DataFrame with compile-time type checking

### 3. Spark APIs and Abstractions

#### RDD (Low-Level API)
```python
# Creating RDDs
rdd = sc.parallelize([1, 2, 3, 4, 5])
file_rdd = sc.textFile("hdfs://path/to/file")

# Transformations (lazy)
mapped_rdd = rdd.map(lambda x: x * 2)
filtered_rdd = rdd.filter(lambda x: x > 2)

# Actions (trigger execution)
result = mapped_rdd.collect()
count = filtered_rdd.count()
```

#### DataFrame (High-Level API)
```python
# Creating DataFrames
df = spark.read.csv("data.csv", header=True, inferSchema=True)
df = spark.createDataFrame(pandas_df)

# Operations
filtered_df = df.filter(df.age > 21)
grouped_df = df.groupBy("department").agg({"salary": "avg"})
result_df = df.select("name", "age").where(df.city == "NYC")
```

#### Spark SQL
```python
# Register DataFrame as table
df.createOrReplaceTempView("employees")

# SQL queries
result = spark.sql("""
    SELECT department, AVG(salary) as avg_salary,
           COUNT(*) as employee_count
    FROM employees
    WHERE age > 25
    GROUP BY department
    ORDER BY avg_salary DESC
""")
```

### 4. Spark Ecosystem

#### Spark Core
- RDD API and basic functionality
- Task scheduling and execution
- Memory management
- Fault tolerance

#### Spark SQL
- DataFrame and Dataset APIs
- SQL query engine
- Schema inference
- Integration with Hive, Parquet, JSON, etc.

#### Spark Streaming
- Real-time data processing
- Micro-batch and continuous processing
- Integration with Kafka, Flume, etc.
- Windowed operations

#### MLlib (Machine Learning)
- Classification, regression, clustering
- Feature engineering
- Model evaluation
- Pipeline API

#### GraphX (Graph Processing)
- Graph computation
- PageRank, connected components
- Graph algorithms
- Integration with GraphFrames

### 5. Performance and Optimization

#### Catalyst Optimizer
- Query optimization engine
- Cost-based optimization
- Logical and physical plan optimization
- Rule-based and cost-based optimizations

#### Tungsten Execution Engine
- Whole-stage code generation
- CPU cache-aware computation
- Reduced memory usage
- Faster serialization

#### Memory Management
- On-heap vs off-heap memory
- Memory fractions (execution, storage, user)
- Spill to disk when memory insufficient
- Garbage collection tuning

#### Caching and Persistence
```python
# Cache levels
df.cache()  # MEMORY_ONLY
df.persist(StorageLevel.MEMORY_AND_DISK)
df.persist(StorageLevel.MEMORY_ONLY_SER)

# Unpersist
df.unpersist()
```

### 6. Data Sources and Formats

#### File Formats
- **Parquet:** Columnar storage, compression, predicate pushdown
- **ORC:** Optimized Row Columnar, Hive integration
- **Avro:** Schema evolution, cross-language compatibility
- **JSON/CSV:** Text formats with schema inference

#### Data Sources
- **HDFS:** Distributed file system
- **S3:** Cloud object storage
- **JDBC:** Relational databases
- **NoSQL:** Cassandra, MongoDB, Elasticsearch
- **Kafka:** Real-time streaming

### 7. Deployment and Operations

#### Cluster Managers
- **Standalone:** Simple cluster manager included with Spark
- **YARN:** Hadoop resource manager integration
- **Kubernetes:** Container orchestration
- **Mesos:** General-purpose cluster manager

#### Configuration
- **Application properties:** spark.app.name, spark.master
- **Runtime properties:** spark.driver.memory, spark.executor.memory
- **Environment variables:** SPARK_HOME, JAVA_HOME
- **Logging:** log4j.properties configuration

#### Monitoring
- **Spark UI:** Web interface for job monitoring
- **Metrics:** System and application metrics
- **Event logging:** Job execution details
- **Ganglia/Graphite:** External monitoring integration

### 8. Best Practices

#### DataFrame vs RDD
- **Use DataFrame when possible:** Higher-level API, optimization
- **Use RDD for:** Custom partitioning, low-level control
- **Mix APIs:** DataFrame for ETL, RDD for complex algorithms

#### Performance Tuning
- **Partitioning:** Right number of partitions (2-4x cores)
- **Caching:** Cache frequently used DataFrames
- **Broadcast joins:** For small tables
- **Coalesce vs repartition:** Minimize shuffling

#### Memory Management
- **Executor memory:** 75% of available RAM
- **Driver memory:** Sufficient for collecting results
- **Storage memory:** Balance with execution memory
- **GC tuning:** Parallel GC for large heaps

### 9. Common Patterns and Use Cases

#### ETL Pipelines
```python
# Extract
raw_df = spark.read.csv("input/*.csv", header=True)

# Transform
cleaned_df = raw_df \
    .filter(col("status") == "active") \
    .withColumn("processed_date", current_date()) \
    .dropDuplicates(["id"])

# Load
cleaned_df.write.parquet("output/processed_data")
```

#### Machine Learning Pipeline
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier

# Feature engineering
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Pipeline
pipeline = Pipeline(stages=[indexer, assembler, rf])
model = pipeline.fit(train_df)
predictions = model.transform(test_df)
```

#### Streaming Analytics
```python
from pyspark.sql.functions import window

# Streaming query
streaming_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .load()

# Windowed aggregation
windowed_counts = streaming_df \
    .groupBy(window(col("timestamp"), "10 minutes"), col("event_type")) \
    .count()

# Output to console
query = windowed_counts \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()
```

### 10. Integration with Big Data Ecosystem

#### Hadoop Integration
- **HDFS:** Storage layer
- **YARN:** Resource management
- **Hive:** SQL interface
- **HBase:** NoSQL database

#### Cloud Platforms
- **AWS EMR:** Managed Spark clusters
- **Google Dataproc:** Cloud Spark service
- **Azure HDInsight:** Big data analytics
- **Databricks:** Unified analytics platform

#### Data Warehousing
- **Delta Lake:** ACID transactions on data lakes
- **Apache Iceberg:** Table format for large datasets
- **Hudi:** Incremental processing framework

## Key Takeaways

1. **Spark is the modern MapReduce:** Faster, more flexible, richer ecosystem
2. **DataFrames are preferred:** Higher-level API with automatic optimization
3. **Memory is key:** In-memory processing enables interactive analytics
4. **Unified platform:** One framework for batch, streaming, ML, and graph processing
5. **Performance matters:** Understanding optimization techniques is crucial
6. **Ecosystem integration:** Works seamlessly with existing big data tools

## Connection to Course Labs

- **Module 10:** MapReduce foundations
- **Module 11 Lab:** PySpark DataFrame operations
- **Module 12:** Spark SQL and Hive integration
- **Module 13:** Advanced analytics with Spark

## Further Reading

- "Learning Spark" (Holden Karau et al.)
- "Spark: The Definitive Guide" (Bill Chambers & Matei Zaharia)
- "High Performance Spark" (Holden Karau & Rachel Warren)
- Spark documentation: https://spark.apache.org/docs/latest/

---

*These notes provide the theoretical foundation for understanding Apache Spark. The hands-on demos and labs will reinforce these concepts through practical implementation.*