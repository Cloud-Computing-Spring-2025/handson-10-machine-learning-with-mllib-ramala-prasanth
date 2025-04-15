# handson-10-MachineLearning-with-MLlib.

#  Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. You will preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.

---



Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

---

##  Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

##  Tasks

### Task 1: Data Preprocessing and Feature Engineering

**Objective:**  
Clean the dataset and prepare features for ML algorithms.

**Steps:**
1. Fill missing values in `TotalCharges` with 0.
2. Encode categorical features using `StringIndexer` and `OneHotEncoder`.
3. Assemble numeric and encoded features into a single feature vector with `VectorAssembler`.

**Code Output:**

```
+--------------------+-----------+
|features            |ChurnIndex |
+--------------------+-----------+
|[0.0,12.0,29.85,29...|0.0        |
|[0.0,1.0,56.95,56....|1.0        |
|[1.0,5.0,53.85,108...|0.0        |
|[0.0,2.0,42.30,184...|1.0        |
|[0.0,8.0,70.70,151...|0.0        |
+--------------------+-----------+
```
---

### Task 2: Train and Evaluate Logistic Regression Model

**Objective:**  
Train a logistic regression model and evaluate it using AUC (Area Under ROC Curve).

**Steps:**
1. Split dataset into training and test sets (80/20).
2. Train a logistic regression model.
3. Use `BinaryClassificationEvaluator` to evaluate.

**Code Output Example:**
```
Logistic Regression Model Accuracy: 0.83
```

---

###  Task 3: Feature Selection using Chi-Square Test

**Objective:**  
Select the top 5 most important features using Chi-Square feature selection.

**Steps:**
1. Use `ChiSqSelector` to rank and select top 5 features.
2. Print the selected feature vectors.

**Code Output Example:**
```
+--------------------+-----------+
|selectedFeatures    |ChurnIndex |
+--------------------+-----------+
|[0.0,29.85,0.0,0.0...|0.0        |
|[1.0,56.95,1.0,0.0...|1.0        |
|[0.0,53.85,0.0,1.0...|0.0        |
|[1.0,42.30,0.0,0.0...|1.0        |
|[0.0,70.70,0.0,1.0...|0.0        |
+--------------------+-----------+

```

---

### Task 4: Hyperparameter Tuning and Model Comparison

**Objective:**  
Use CrossValidator to tune models and compare their AUC performance.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Steps:**
1. Define models and parameter grids.
2. Use `CrossValidator` for 5-fold cross-validation.
3. Evaluate and print best model results.

**Code Output Example:**
```
Tuning LogisticRegression...
LogisticRegression Best Model Accuracy (AUC): 0.84
Best Params for LogisticRegression: regParam=0.01, maxIter=20

Tuning DecisionTree...
DecisionTree Best Model Accuracy (AUC): 0.77
Best Params for DecisionTree: maxDepth=10

Tuning RandomForest...
RandomForest Best Model Accuracy (AUC): 0.86
Best Params for RandomForest: maxDepth=15
numTrees=50

Tuning GBT...
GBT Best Model Accuracy (AUC): 0.88
Best Params for GBT: maxDepth=10
maxIter=20

```
---

##  Execution Instructions

### 1. Prerequisites

- Apache Spark installed
- Python environment with `pyspark` installed
- `customer_churn.csv` placed in the project directory

### 2. Run the dataset generation file

```
python dataset-generator.py
```

### 2. Run the python file

```bash
spark-submit customer-churn-analysis.py
```
### Output is included in the text files.

🧹 Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
1.1 Fill Missing Values
```
df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull() | (col("TotalCharges") == ""), 0).otherwise(col("TotalCharges").cast("double")))
```
Replaces null or blank TotalCharges with 0 and casts to numeric.

1.2 Encode Label Column
```
label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
df = label_indexer.fit(df).transform(df)
```
Converts Churn column from "Yes"/"No" to 1/0 for binary classification.

1.3 Index and One-Hot Encode Categorical Features
```
categorical_cols = ["gender", "PhoneService", "InternetService"]
...
```
Each categorical feature is transformed via:

StringIndexer → assigns index values to categories.

OneHotEncoder → converts indexed values into binary vector format.

1.4 Assemble All Features
```
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
final_df = assembler.transform(df)
```
Combines all numeric columns and encoded vectors into a single features vector required by Spark ML models.

🤖 Task 2: Train and Evaluate Logistic Regression
```
def train_logistic_regression_model(df):
```
2.1 Split Data
```
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
```
Splits data into 80% training and 20% test sets.

2.2 Train Logistic Regression
```
lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(train_df)
```
Trains a logistic regression classifier on the training data.

2.3 Evaluate Model
```
predictions = model.transform(test_df)
...
auc = evaluator.evaluate(predictions)
```
Evaluates model performance using AUC (Area Under ROC Curve).

🧪 Task 3: Feature Selection using Chi-Square Test
```
def feature_selection(df):
```
Uses ChiSqSelector to identify top 5 features most associated with the target label.

```
selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", labelCol="label", outputCol="selectedFeatures")
```
Outputs a new DataFrame with the reduced selectedFeatures vector.

🎯 Task 4: Hyperparameter Tuning & Model Comparison
```
def tune_and_compare_models(df):
```
4.1 Define Models and Parameter Grids
Four models are defined:

LogisticRegression

DecisionTreeClassifier

RandomForestClassifier

GBTClassifier (Gradient Boosted Trees)
```
paramGrids = {
    "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).build(),
    ...
}
```
4.2 CrossValidator Setup
```
cv = CrossValidator(estimator=model, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
```
For each model, 5-fold cross-validation is performed to find the best hyperparameters and compute AUC.

4.3 Compare and Report
AUC scores and best parameters for each model are written to task4_output.txt.

📝 Output Writing
Each task appends its results to a corresponding .txt file for recordkeeping:

```
with open("taskX_output.txt", "a") as f:
    f.write(...)
```
```
spark.stop()
```
Closes the Spark session after all tasks complete.

