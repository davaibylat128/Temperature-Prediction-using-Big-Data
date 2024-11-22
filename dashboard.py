import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, to_date
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.feature import StringIndexerModel, VectorAssembler

# Set up environment for Spark
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("TemperaturePredictionAnalysis") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020") \
    .getOrCreate()

# Paths for saved models and preprocessed data in HDFS
model_path = "hdfs://namenode:8020/output/linear_regression_model"
indexers_path = "hdfs://namenode:8020/output/indexers"
preprocessed_data_path = "hdfs://namenode:8020/output/preprocessed_data"

# Load the saved models
gbt_model = GBTRegressionModel.load(model_path)
region_indexer = StringIndexerModel.load(f"{indexers_path}/region_indexer")
country_indexer = StringIndexerModel.load(f"{indexers_path}/country_indexer")
city_indexer = StringIndexerModel.load(f"{indexers_path}/city_indexer")
season_indexer = StringIndexerModel.load(f"{indexers_path}/season_indexer")

# Streamlit App UI
st.title("Temperature Prediction and Analysis App")
st.write("Predict average temperatures and explore climate trends using visualizations.")

# Load preprocessed data
df = spark.read.parquet(preprocessed_data_path)
df = df.drop("features")  
pandas_df = df.limit(100).toPandas()  

# Tabbed UI for Prediction and Visualization
tab1, tab2 = st.tabs(["Temperature Prediction", "Data Analysis"])

# Tab 1: Temperature Prediction
with tab1:
    st.header("Predict Temperature")
    
    # User Inputs
    region = st.text_input("Region (e.g., Asia)", "")
    country = st.text_input("Country (e.g., India)", "")
    city = st.text_input("City (e.g., Mumbai)", "")
    wind = st.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
    precipitation = st.number_input("Precipitation (mm)", min_value=0.0, step=0.1)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
    year = st.number_input("Year (e.g., 2023)", min_value=1900, max_value=2100, step=1)
    day = st.number_input("Day", min_value=1, max_value=31, step=1)
    month = st.number_input("Month", min_value=1, max_value=12, step=1)

    # Prediction Button
    if st.button("Predict Temperature"):
        if not region or not country or not city:
            st.error("Please fill out the required fields!")
        else:
            try:
                input_data = [(region, country, city, wind, precipitation, season, year, month, day)]
                columns = ["region", "country", "city", "wind", "precipitation", "season", "year", "month", "day"]
                input_df = spark.createDataFrame(input_data, columns)

                # Transform input using indexers
                input_df = region_indexer.transform(input_df)
                input_df = country_indexer.transform(input_df)
                input_df = city_indexer.transform(input_df)
                input_df = season_indexer.transform(input_df)

                # Assemble Features
                assembler = VectorAssembler(
                    inputCols=["region_index", "country_index", "city_index", "wind", "precipitation", "season_index", "year", "month", "day"],
                    outputCol="features"
                )
                input_df = assembler.transform(input_df)

                # Make Prediction
                predictions = gbt_model.transform(input_df)
                predicted_temp = predictions.select("prediction").first()["prediction"]

                st.success(f"The predicted average temperature is: {predicted_temp:.2f} °F")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Tab 2: Data Analysis
with tab2:
    st.header("Explore Data and Visualizations")
    
    pandas_df = df.toPandas()  # Limit rows for performance
    st.write("Sample Data:")
    st.dataframe(pandas_df)

    # Distribution of Average Temperature
    st.subheader("1. Distribution of Avg Temperature")
    fig, ax = plt.subplots()
    sns.histplot(pandas_df["avgtemperature"], kde=True, bins=30, ax=ax, color="skyblue")
    ax.set_title("Distribution of Avg Temperature")
    st.pyplot(fig)

    # Scatter Plot: Avg Temperature vs Wind Speed
    st.subheader("2. Avg Temperature vs Wind Speed")
    fig, ax = plt.subplots()
    sns.scatterplot(x="wind", y="avgtemperature", data=pandas_df, ax=ax, alpha=0.7)
    ax.set_title("Scatter Plot: Avg Temperature vs Wind Speed")
    st.pyplot(fig)

    # Avg Temperature by Season
    st.subheader("3. Avg Temperature by Season")
    if "season" in pandas_df.columns:
        grouped_data = pandas_df.groupby("season")["avgtemperature"].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x="season", y="avgtemperature", data=grouped_data, palette="coolwarm", ax=ax)
        ax.set_title("Avg Temperature by Season")
        st.pyplot(fig)


   

    st.subheader("4. Seasonal Trends by Year")
    if "season" in pandas_df.columns and "year" in pandas_df.columns:
        # Filter out years below 1995
        filtered_data = pandas_df[pandas_df["year"] >= 1995]
    
    # Group by year and season, then calculate the average temperature
        seasonal_data = filtered_data.groupby(["year", "season"])["avgtemperature"].mean().reset_index()
    
    # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=seasonal_data, x="year", y="avgtemperature", hue="season", marker="o", ax=ax)
        ax.set_title("Seasonal Temperature Trends by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Temperature (°F)")
        st.pyplot(fig)
    else:
        st.warning("Season or Year column is missing for this analysis.")


     # Regional Variation of Temperature
    st.subheader("5. Regional Variation of Temperature")
    if "region" in pandas_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x="region", y="avgtemperature", data=pandas_df, ax=ax, palette="coolwarm")
        ax.set_title("Regional Variation of Temperature")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("6. Correlation Heatmap")
    correlation_data = pandas_df[["avgtemperature","region_index", "country_index", "city_index", "wind", "precipitation","month","day","year","season_index"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_data, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# Stop Spark Session
if st.button("Stop Spark Session"):
    spark.stop()
    st.success("Spark session stopped.")
