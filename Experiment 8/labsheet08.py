# ============================================================
# EXPERIMENT NO. 8
# Customer Churn Prediction using Decision Tree Classification
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.tree import plot_tree


# ============================================================
# 2. LOAD DATASET
# ============================================================

# IMPORTANT:
# Use r before the Windows path
file_path = "D:\machine learning\cad lab\WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Check if file exists
if not os.path.exists(file_path):
    print("ERROR: Dataset file not found!")
    print("\nPlease check the file path:")
    print(file_path)
    exit()

# Load CSV file
df = pd.read_csv(file_path)

print("\n================================================")
print("DATASET LOADED SUCCESSFULLY")
print("================================================")

print("\nFirst 5 rows:")
print(df.head())
print("priyanshu sharma cu240250980")

# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\n================================================")
print("DATASET INFORMATION")
print("================================================")

print("\nNumber of rows and columns:")
print(df.shape)
print("priyanshu sharma cu240250980")
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates()

# Convert TotalCharges into numeric
# Some rows may contain blank spaces
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values after conversion
print("\nMissing values after converting TotalCharges:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

print("\nDataset shape after preprocessing:")
print(df.shape)


# ============================================================
# 5. REMOVE CUSTOMER ID
# ============================================================

# CustomerID does not help in predicting churn
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)


# ============================================================
# 6. ENCODE CATEGORICAL VARIABLES
# ============================================================

# Convert Yes/No columns and other categorical columns
# into numerical values using one-hot encoding.

df = pd.get_dummies(
    df,
    columns=df.select_dtypes(include=["object"]).columns,
    drop_first=True
)

print("\n================================================")
print("DATA AFTER ENCODING")
print("================================================")
print("priyanshu sharma cu240250980")
print(df.head())

print("\nNumber of columns after encoding:")
print(df.shape[1])


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

# Churn_Yes is the target column after encoding

if "Churn_Yes" not in df.columns:
    print("\nERROR: Churn_Yes column not found!")
    print("Available columns:")
    print(df.columns.tolist())
    exit()

X = df.drop("Churn_Yes", axis=1)

y = df["Churn_Yes"]


print("\n================================================")
print("FEATURES AND TARGET")
print("================================================")

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("priyanshu sharma cu240250980")

# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n================================================")
print("TRAIN-TEST SPLIT")
print("================================================")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 9. CREATE DECISION TREE CLASSIFIER
# ============================================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)


# ============================================================
# 10. TRAIN THE MODEL
# ============================================================

model.fit(X_train, y_train)

print("\n================================================")
print("DECISION TREE MODEL TRAINED")
print("================================================")


# ============================================================
# 11. PREDICT CUSTOMER CHURN
# ============================================================

y_pred = model.predict(X_test)

print("\nPredicted values:")
print(y_pred[:20])
print("priyanshu sharma cu240250980")

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n================================================")
print("CONFUSION MATRIX")
print("================================================")

print(cm)


# Visualize confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n================================================")
print("MODEL PERFORMANCE")
print("================================================")

print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1 Score  :", round(f1, 4))
print("priyanshu sharma cu240250980")

# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )
)


# ============================================================
# 15. VISUALIZE DECISION TREE
# ============================================================

plt.figure(figsize=(25, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree for Customer Churn Prediction")

plt.tight_layout()
plt.show()


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

# Sort features by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n================================================")
print("FEATURE IMPORTANCE")
print("================================================")

print(feature_importance.head(15))


# ============================================================
# 17. VISUALIZE TOP 10 IMPORTANT FEATURES
# ============================================================

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# ============================================================
# 18. ACTUAL VS PREDICTED CHURN
# ============================================================

result = pd.DataFrame({
    "Actual_Churn": y_test.values,
    "Predicted_Churn": y_pred
})

print("\n================================================")
print("ACTUAL VS PREDICTED CHURN")
print("================================================")

print(result.head(20))


# ============================================================
# 19. SAVE PREDICTION RESULTS
# ============================================================

result.to_csv(
    "churn_prediction_results.csv",
    index=False
)


# ============================================================
# 20. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)


# ============================================================
# 21. FINAL RESULT
# ============================================================

print("\n================================================")
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("================================================")

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")
print("priyanshu sharma cu240250980")
print("\nFiles created:")
print("1. churn_prediction_results.csv")
print("2. feature_importance.csv")

print("\nDecision Tree Churn Prediction completed!")