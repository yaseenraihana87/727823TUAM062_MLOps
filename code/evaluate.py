# Yaseen Raihana - 727823TUAM062

import pandas as pd
import time
import joblib
from sklearn.metrics import f1_score

print("Roll No: 727823TUAM062")
print("Timestamp:", time.strftime("%Y-%m-%d %H:%M:%S"))

df = pd.read_csv("processed_data.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = joblib.load("model.pkl")

preds = model.predict(X)

f1 = f1_score(y, preds, average="weighted")

print("F1 Score:", f1)