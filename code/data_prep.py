# Yaseen Raihana - 727823TUAM062

import pandas as pd
import time

print("Roll No: 727823TUAM062")
print("Timestamp:", time.strftime("%Y-%m-%d %H:%M:%S"))

df = pd.read_csv("code/data/steel_faults.csv")

# Simple preprocessing
df = df.dropna()

df.to_csv("processed_data.csv", index=False)

print("Data preprocessing completed")