#Importing all the Libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA



# Reading the dataset
df = pd.read_csv("/content/customer_segmentation_data.csv")
columns = df.columns



# Checking numbers of rows and columns in dataset
print("Rows, Columns:", df.shape)



#Checking for null values
df.isnull().sum()
print("\n\n\n")


# Remove duplicate rows if any & also drpoing Customer ID column 

df = df.drop_duplicates()
df = df.dropna()








###### EDA ######



# Chart for Gender 
plt.figure(figsize=(6,4))
sns.countplot(x="gender", data=df)
plt.xlabel('Other-Gender', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title("Gender Distribution of Customers")
plt.show()
print("\n\n\n")


# Chart for Age Distribution
plt.figure(figsize=(10, 6))
p = sns.histplot(df['age'], kde=True, bins=30, alpha=0.5, fill=True)
p.axes.set_title("\nCustomer's Age Distribution\n",fontsize=25)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Count', fontsize=15)
sns.despine(left=True, bottom=True)
plt.show()
print("\n\n\n")


# Chart for Income Distribution
plt.figure(figsize=(10, 6))
p = sns.histplot(data=df, x='income', kde=True, bins=30, alpha=1, fill=True, hue= 'gender')
p.axes.set_title("\nCustomer's Income Distribution\n",fontsize=25)
plt.xlabel('Income', fontsize=15)
plt.ylabel('Count', fontsize=15)
sns.despine(left=True, bottom=True)
plt.show()
print("\n\n\n")




# Chart for Spending Score Distribution
plt.figure(figsize=(10, 6))
p = sns.histplot(df['spending_score'], kde=True, bins=30, alpha=0.5, fill=True)
p.axes.set_title("\nCustomer Spending Score Distribution\n",fontsize=25)
plt.xlabel('Spending Score', fontsize=15)
plt.ylabel('Count', fontsize=15)
sns.despine(left=True, bottom=True)
plt.show()
print("\n\n\n")




# Chart for Membership Score Distribution
plt.figure(figsize=(10, 6))
p = sns.histplot(df['membership_years'], kde=True, bins=30, alpha=0.5, fill=True)
p.axes.set_title("\nCustomer Membership Score Distribution\n",fontsize=25)
plt.xlabel('Membership-Years', fontsize=15)
plt.ylabel('Count', fontsize=15)
sns.despine(left=True, bottom=True)
plt.show()
print("\n\n\n")





# Chart for Last Purchase Distribution
plt.figure(figsize=(10, 8))
p = sns.histplot(data=df, x='last_purchase_amount', kde=True, edgecolor='white', bins=30, alpha=1, fill=True, hue= 'preferred_category')
p.axes.set_title("\nCustomer's Last Purchase Distribution\n",fontsize=25)
plt.xlabel('Aamount', fontsize=15)
plt.ylabel('Count', fontsize=15)
sns.despine(left=True, bottom=True)
plt.show()
print("\n\n\n")





# Heatmap
df_cluster = df.select_dtypes(include=["int64", "float64"])
df_cluster = df_cluster.drop(columns=["id"], errors="ignore")

plt.figure(figsize=(8,6))
sns.heatmap(df_cluster.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation (Numerical Features Only)")
plt.show()
print("\n\n\n")




# Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)





# Finding optimal number of clusters

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)




# Ploting of Optical K Method

plt.figure(figsize=(6,4))
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.show()
print("\n\n\n")




# Applying K Means Cluster

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

sil_score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", sil_score)
print("\n\n\n")




# Cluster Profiling

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

cluster_profile = df.groupby("Cluster")[numeric_cols].mean()
cluster_profile
print("\n\n\n")





# Visualizing Clusters

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

plt.figure(figsize=(6,5))
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    data=df,
    palette="Set2"
)
plt.title("Customer Segments Visualization (PCA)")
plt.show()
print("\n\n\n")



df.to_csv("/content/customer_segmentation_data.csv", index=False)

