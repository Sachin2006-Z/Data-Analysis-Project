# Importing all the Libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Reading the Da6aset

df = pd.read_csv(
    "/content/FA-KES-Dataset.csv",
    encoding="latin1"
)


#  Adding Column Article title and content 

df["text"] = (
    df["article_title"].astype(str) + " " +
    df["article_content"].astype(str)
)



# For normalizing the Labels

df["labels"] = (
    df["labels"]
    .astype(str)
    .str.upper()
    .str.strip()
)

print(df["labels"].value_counts())




# Ensure labels are numeric
df["labels"] = pd.to_numeric(df["labels"], errors="coerce")



# Droping invalid rows (if any)

df.dropna(subset=["labels"], inplace=True)
df["labels"] = df["labels"].astype(int)




# EDA 

plt.figure(figsize=(6,4))
sns.countplot(x="labels", data= df)
plt.title("Fake & Real")
plt.xlabel(" Fake=0 & Real=1")
plt.show()




# Converting Text to Numerics for the testing

texts = df["text"].astype(str).values
y = df["labels"].values

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)




# Testing & Training
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# Class Weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)



# Tenserflow

model = Sequential([
    Embedding(
        input_dim=10000,   
        output_dim=64,
        input_length=100
    ),
    GlobalAveragePooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)




# Training the model

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weight_dict,
    verbose=1
)






# Evaluating the data

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))




# Making the function of pridiction
def predict_news(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)[0][0]
    return "Real News" if pred > 0.5 else "Fake News"


# Final test
print(predict_news("Government announces new education policy for students"))
print(predict_news("Aliens found living secretly under the White House"))


