# ============================================================
# EXPERIMENT NO. 7
# Customer Segmentation using K-Means Clustering
# ============================================================

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
 1. LOAD DATASET
# ============================================================

# IMPORTANT:
# r before# the string prevents Windows path errors
file_path = r"D:\machine learning\cad lab\Mall_Customers.csv"

# Check whether file exists
if not os.path.exists(file_path):
    print("ERROR: Dataset file not found!")
    print("Please check this path:")
    print(file_path)
    exit()

# Read dataset
df = pd.read_csv(file_path)

print("\n================ DATASET ================\n")
print(df.head())


# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("\n================ DATASET INFORMATION ================\n")
df.info()

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows containing missing values
df = df.dropna()

print("\n================ AFTER PREPROCESSING ================\n")
print("Dataset shape:", df.shape)


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

for column in required_columns:
    if column not in df.columns:
        print("\nERROR: Column not found:", column)
        print("\nAvailable columns are:")
        print(df.columns.tolist())
        exit()


# ============================================================
# 5. SELECT FEATURES
# ============================================================

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

print("\n================ SELECTED FEATURES ================\n")
print(X.head())


# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n================ SCALED DATA ================\n")
print(X_scaled[:5])


# ============================================================
# 7. ELBOW METHOD
# ============================================================

inertia = []

K_range = range(1, 11)

for k in K_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


# Display Elbow graph
plt.figure(figsize=(8, 5))

plt.plot(
    K_range,
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(K_range)
plt.grid(True)

plt.show()


# ============================================================
# 8. SILHOUETTE SCORE
# ============================================================

silhouette_scores = []

K_range_silhouette = range(2, 11)

print("\n================ SILHOUETTE SCORES ================\n")
print("priyanshu sharma cu240250980")
for k in K_range_silhouette:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        "K =", k,
        "Silhouette Score =", round(score, 4)
    )


# Display Silhouette graph
plt.figure(figsize=(8, 5))

plt.plot(
    K_range_silhouette,
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(K_range_silhouette)
plt.grid(True)

plt.show()


# ============================================================
# 9. SELECT OPTIMAL K
# ============================================================

best_k = K_range_silhouette[
    np.argmax(silhouette_scores)
]

print("\n==========================================")
print("Optimal Number of Clusters:", best_k)
print("==========================================")
print("priyanshu sharma cu240250980")

# ============================================================
# 10. APPLY K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# ============================================================
# 11. CLUSTER CENTROIDS
# ============================================================

centroids_scaled = kmeans.cluster_centers_

# Convert scaled centroids back to original values
centroids = scaler.inverse_transform(
    centroids_scaled
)

centroid_df = pd.DataFrame(
    centroids,
    columns=features
)

print("\n================ CLUSTER CENTROIDS ================\n")
print(centroid_df)


# ============================================================
# 12. DISPLAY CUSTOMER CLUSTERS
# ============================================================

print("\n================ CUSTOMER CLUSTERS ================\n")

display_columns = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Cluster"
]

# Add these columns only if they exist
if "CustomerID" in df.columns:
    display_columns.insert(0, "CustomerID")

if "Gender" in df.columns:
    display_columns.insert(1, "Gender")

print(df[display_columns].head(20))
print("priyanshu sharma cu240250980")

# ============================================================
# 13. CLUSTER SIZE
# ============================================================

print("\n================ CLUSTER SIZE ================\n")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_counts)


# ============================================================
# 14. CLUSTER ANALYSIS
# ============================================================

cluster_summary = df.groupby("Cluster")[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()

print("\n================ CLUSTER SUMMARY ================\n")
print("priyanshu sharma cu240250980")
print(cluster_summary)


# ============================================================
# 15. VISUALIZE CLUSTERS
# ============================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

# Plot cluster centroids
plt.scatter(
    centroids[:, 1],
    centroids[:, 2],
    marker="X",
    s=300,
    c="red",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 16. AGE VS SPENDING SCORE
# ============================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="Age",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Age vs Spending Score by Customer Cluster")
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 17. SAVE RESULT
# ============================================================

output_file = r"D:\machine learning\cad lab\customer_segmentation_result.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n==========================================")
print("Clustering completed successfully!")
print("Result saved at:")
print(output_file)
print("priyanshu sharma cu240250980")
print("==========================================")