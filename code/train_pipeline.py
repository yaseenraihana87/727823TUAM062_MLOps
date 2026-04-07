# Yaseen Raihana - 727823TUAM062

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Roll No: 727823TUAM062")
print("Timestamp:", time.strftime("%Y-%m-%d %H:%M:%S"))

df = pd.read_csv("processed_data.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model training completed")