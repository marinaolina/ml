# Databricks notebook source
df = spark.read.option("header", "true").option("inferSchema", "true").csv("/FileStore/tables/Impact_of_Remote_Work_on_Mental_Health.csv")

# COMMAND ----------

df.display()

# COMMAND ----------

df.columns

# COMMAND ----------

df1 = df.select("Satisfaction_with_Remote_Work").distinct().display()

# COMMAND ----------

# MAGIC %md
# MAGIC  - dataset is relatively small
# MAGIC  - classes are spread evenly
# MAGIC  - contains numerical and categorical columns

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


categorical_columns = ['Satisfaction_with_Remote_Work', 'Gender', 'Job_Role', 'Industry', 'Work_Location', 'Mental_Health_Condition', 'Access_to_Mental_Health_Resources', 'Region', 'Stress_Level', 'Productivity_Change', 'Physical_Activity', 'Sleep_Quality']
indexed_cols = indexed_columns = [col + '_indexed' for col in categorical_columns]
numerical_columns = ['Age', 'Years_of_Experience', 'Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating', 'Social_Isolation_Rating', 'Company_Support_for_Remote_Work']

indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_indexed")
    for col in categorical_columns
]

pipeline = Pipeline(stages=indexers)

model = pipeline.fit(df)
transformed_df = model.transform(df)

transformed_df.display()

# COMMAND ----------

df_num = transformed_df.select("Employee_ID",*numerical_columns, *indexed_cols).cache()
df_num.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Steps:
# MAGIC Identify Feature Columns:
# MAGIC
# MAGIC Exclude Employee_ID (identifier) and the target column Satisfaction_with_Remote_Work.
# MAGIC Use all other columns as features for factor analysis.
# MAGIC VectorAssembler:
# MAGIC
# MAGIC Combine all feature columns into a single feature vector.
# MAGIC OneHotEncoding:
# MAGIC
# MAGIC If any columns need one-hot encoding, ensure they are already transformed to numerical or sparse vectors.
# MAGIC Modeling:
# MAGIC
# MAGIC Use factor analysis tools, such as PCA, if applicable, or a regression model to evaluate the impact of features on the target.
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.stat import Correlation

feature_columns = [col for col in df_num.columns if col not in ["Employee_ID", "Satisfaction_with_Remote_Work_indexed"]]
target_column = "Satisfaction_with_Remote_Work_indexed"

# Single vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

assembled_df = assembler.transform(df_num)
assembled_df.display()

# COMMAND ----------

from pyspark.sql import functions as F
final_df = assembled_df.select("features", F.col(target_column).alias("label"))
final_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC > linear and andom forest gave unstable results (at each change of see changes the range of factors)

# COMMAND ----------

train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=987)

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor

# Configure GBT Regressor
gbt = GBTRegressor(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    maxIter=50,          # Number of boosting iterations
    maxDepth=5,          # Maximum depth of each tree
    seed=987              # Set seed for reproducibility
)

# Fit the model
gbt_model = gbt.fit(train_df)

# Extract feature importances
importances = [float(i) for i in gbt_model.featureImportances]

# Create a DataFrame with feature importances
importance_df = spark.createDataFrame(zip(feature_columns, importances), ["Feature", "Importance"])

# Show top features
importance_df.orderBy(F.col("Importance").desc()).display()


# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

# Predict on the test set
predictions = gbt_model.transform(test_df)

# Evaluate RMSE on test data
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) with Important Features: {rmse:.2f}")

# COMMAND ----------

from pyspark.sql import functions as F

# Use the trained model to predict on the dataset
predictions = gbt_model.transform(final_df)

# Add a column to compare actual and predicted satisfaction
results = predictions.select(
    F.col("label").alias("Actual_Satisfaction"),       # Actual satisfaction values
    F.col("prediction").alias("Predicted_Satisfaction")  # Predicted satisfaction values
)

# Add a column for rounded predictions (if satisfaction is categorical)
results = results.withColumn(
    "Rounded_Predicted_Satisfaction", F.round(F.col("Predicted_Satisfaction")).cast("int")
)

# Add a column to check if the prediction is correct
results = results.withColumn(
    "Correct_Prediction",
    F.when(F.col("Rounded_Predicted_Satisfaction") == F.col("Actual_Satisfaction"), 1).otherwise(0)
)

# Show the comparison
results.display()




# COMMAND ----------

# Calculate accuracy
total_predictions = results.count()
correct_predictions = results.filter(F.col("Correct_Prediction") == 1).count()
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy:.2f}")

