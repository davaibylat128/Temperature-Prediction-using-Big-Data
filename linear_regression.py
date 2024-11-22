# Import necessary libraries
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, concat, lit, to_date, avg
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

# Environment setup
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

# Step 1: Start SparkSession
spark = SparkSession.builder \
    .appName("LinearRegressionWithHDFS") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020") \
    .getOrCreate()

# Step 2: Load the dataset from HDFS
df = spark.read.csv("hdfs://namenode:8020/input/data/city_temperature.csv", header=True, inferSchema=True)

# Step 3: Basic Dataset Info
num_columns = len(df.columns)
num_rows = df.count()
print(f"Dataset contains {num_columns} columns and {num_rows} rows.")

# Step 4: Check for Nulls
null_counts = df.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in df.columns])
null_counts.show()


# Step 5: Handle Missing Values
mean_temp = df.filter(col("avgtemperature") != -99).select(avg("avgtemperature").alias("mean_temp")).collect()[0]["mean_temp"]
df = df.withColumn("avgtemperature", when(col("avgtemperature") == -99, mean_temp).otherwise(col("avgtemperature")))

# Step 6: Drop Unnecessary Columns
df = df.drop("state")

#Removal of inconsistant data
unique_years = df.select("year").distinct()
# Show the unique values
unique_years.show()

#Eg. 200

day_zero = df.filter(df.day == 0)
 # Show the filtered DataFrame
day_zero.show()

year_inconsistent=df.filter(df.year<1995).sample(withReplacement=False, fraction=0.1, seed=42)
# To get a specific number of samples (e.g., 10), we can use limit after filtering
year_inconsistent = df.filter(df.year < 1995).limit(10)
# Show the resulting DataFrame
year_inconsistent.show()




# Step 6: Add 'season' feature
df = df.withColumn(
    "season",
    when((col("month").isin(12, 1, 2)), "Winter")
    .when((col("month").isin(3, 4, 5)), "Spring")
    .when((col("month").isin(6, 7, 8)), "Summer")
    .otherwise("Autumn")
)

# Step 7: Combine day, month, and year into a date column
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
df = df.withColumn("date", to_date(concat(col("year"), lit("-"), col("month"), lit("-"), col("day")), "yyyy-MM-dd"))

# Step 8: Index Categorical Columns
indexers = {
    col_name: StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
    for col_name in ["region", "country", "city", "season"]
}

for col_name, indexer in indexers.items():
    indexers_path = "hdfs://namenode:8020/output/indexers"
    indexer_model = indexer.fit(df)
    df = indexer_model.transform(df)
    indexer_model.write().overwrite().save(f"{indexers_path}/{col_name}_indexer")

# Step 9: Prepare Features for Machine Learning
assembler = VectorAssembler(
    inputCols=["region_index", "country_index", "city_index", "wind", "precipitation","month","day","year","season_index"],
    outputCol="features"
)
df = assembler.transform(df)

# Step 10: Save Preprocessed Data to HDFS
spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "LEGACY")
preprocessed_data_path = "hdfs://namenode:8020/output/preprocessed_data"
df.write.mode("overwrite").parquet(preprocessed_data_path)
print(f"Preprocessed data saved to {preprocessed_data_path}")

# Step 11: Split Data for Training and Testing
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# Step 12: Train Linear Regression Model
gbt = GBTRegressor(featuresCol="features", labelCol="avgtemperature", maxIter=100, maxBins=200)

# Train the model
gbt_model = gbt.fit(train_data)

# Make predictions
gbt_predictions = gbt_model.transform(test_data)

# Initialize evaluator
evaluator = RegressionEvaluator(labelCol="avgtemperature", predictionCol="prediction", metricName="rmse")

# Evaluate the model
gbt_rmse = evaluator.evaluate(gbt_predictions)
print(f"Gradient Boosting RMSE: {gbt_rmse}")

gbt_r2 = evaluator.setMetricName("r2").evaluate(gbt_predictions)
print(f"Gradient Boosting R2: {gbt_r2}")

# Step 15: Show Sample Predictions
gbt_predictions.select("features", "avgtemperature", "prediction").show()

# Step 16: Save the Trained Model
model_path = "hdfs://namenode:8020/output/linear_regression_model"
gbt_model.write().overwrite().save(model_path)
print(f"Model saved to {model_path}")

# Stop the Spark session
spark.stop()
