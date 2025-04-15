from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import sys

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)


# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    with open("task1_output.txt", "w") as f:
        # Fill missing or invalid TotalCharges with 0
        df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges")))

        # Convert target column to numeric
        label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
        df = label_indexer.fit(df).transform(df)

        # Categorical columns
        cat_cols = ["gender", "PhoneService", "InternetService"]
        indexers = [StringIndexer(inputCol=c, outputCol=c+"_Index") for c in cat_cols]
        encoders = [OneHotEncoder(inputCol=c+"_Index", outputCol=c+"_Vec") for c in cat_cols]

        for indexer in indexers:
            df = indexer.fit(df).transform(df)
        for encoder in encoders:
            df = encoder.fit(df).transform(df)

        # Feature columns
        numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
        one_hot_cols = [c + "_Vec" for c in cat_cols]
        feature_cols = numeric_cols + one_hot_cols

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        final_df = assembler.transform(df).select("features", "label")

        f.write("Task 1 completed: Data preprocessed and features assembled.\n")
        f.write(f"Sample row:\n{final_df.limit(1).collect()[0]}\n")

        return final_df


# Task 2: Train and Evaluate a Logistic Regression Model
def train_logistic_regression_model(df):
    with open("task2_output.txt", "w") as f:
        train, test = df.randomSplit([0.8, 0.2], seed=42)
        lr = LogisticRegression()
        model = lr.fit(train)
        predictions = model.transform(test)

        evaluator = BinaryClassificationEvaluator()
        auc = evaluator.evaluate(predictions)
        f.write(f"Logistic Regression AUC: {auc:.4f}\n")


# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    with open("task3_output.txt", "w") as f:
        selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
        result = selector.fit(df).transform(df)

        f.write("Top 5 selected features using Chi-Square Test:\n")
        for row in result.select("selectedFeatures", "label").limit(5).collect():
            f.write(str(row) + "\n")


# Task 4: Hyperparameter Tuning and Model Comparison
def tune_and_compare_models(df):
    with open("task4_output.txt", "w") as f:
        train, test = df.randomSplit([0.8, 0.2], seed=42)
        evaluator = BinaryClassificationEvaluator()

        models = {
            "LogisticRegression": LogisticRegression(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
            "GBT": GBTClassifier()
        }

        param_grids = {
            "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).build(),
            "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [3, 5]).build(),
            "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].numTrees, [10, 20]).build(),
            "GBT": ParamGridBuilder().addGrid(models["GBT"].maxIter, [10, 20]).build()
        }

        best_auc = 0
        best_model_name = None
        best_model = None

        for name in models:
            f.write(f"Training {name}...\n")
            cv = CrossValidator(estimator=models[name],
                                estimatorParamMaps=param_grids[name],
                                evaluator=evaluator,
                                numFolds=5)
            cv_model = cv.fit(train)
            predictions = cv_model.transform(test)
            auc = evaluator.evaluate(predictions)
            f.write(f"{name} AUC: {auc:.4f}\n")

            if auc > best_auc:
                best_auc = auc
                best_model_name = name
                best_model = cv_model.bestModel

        f.write(f"\nBest Model: {best_model_name} with AUC = {best_auc:.4f}\n")
        f.write(f"Best Params: {best_model.extractParamMap()}\n")


# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()
