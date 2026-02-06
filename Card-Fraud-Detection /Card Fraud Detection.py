# Importing Libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix





# Reading the Dataset

df=pd.read_csv("/content/creditcard.csv")





# Dataset Shaping

df.head()
print("Rows, Columns:", df.shape)

df["Class"].value_counts()
print("\n\n\n")





# Value counts

plt.figure(figsize=(5,8))
sns.countplot(x="Class", data=df)
plt.xlabel("Genuine=0 : Fraud=1")
plt.title("Fraud vs Genuine Transactions")
plt.show()
print("\n\n\n")




# Drop rows with missing labels FIRST
df = df.dropna(subset=["Class"])

X = df.drop("Class", axis=1)
y = df["Class"]




# Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)





# Training 

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)





#  Logistic Regression {Model-1}

log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

log_model.fit(X_train, y_train)
print("\n\n")


y_pred_log = log_model.predict(X_test)

print("Logistic Regression Results")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print("\n\n")





# Random Forest {Model-2}

rf_model = RandomForestClassifier(
    n_estimators=20,      
    max_depth=6,          
    min_samples_split=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Results")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("|n\n")






# Logistic Regression metrics
log_precision = precision_score(y_test, y_pred_log)
log_recall = recall_score(y_test, y_pred_log)
log_f1 = f1_score(y_test, y_pred_log)






# Random Forest metrics
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)


metrics_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Precision": [log_precision, rf_precision],
    "Recall": [log_recall, rf_recall],
    "F1-Score": [log_f1, rf_f1]
})

metrics_df






# Ploting of comparision Bar Chart

metrics_df.set_index("Model").plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Model Comparison on Fraud Detection Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()




