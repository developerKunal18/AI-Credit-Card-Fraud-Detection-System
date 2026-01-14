import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample transaction dataset
data = {
    "amount": [50, 5000, 20, 10000, 30, 8000],
    "location_risk": [1, 5, 1, 5, 1, 4],
    "device_risk": [1, 4, 1, 5, 1, 4],
    "fraud": [0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["amount", "location_risk", "device_risk"]]
y = df["fraud"]

model = LogisticRegression()
model.fit(X, y)

print("💳 Credit Card Fraud Detector\n")

amt = float(input("Transaction amount: "))
loc = int(input("Location risk (1–5): "))
dev = int(input("Device risk (1–5): "))

pred = model.predict([[amt, loc, dev]])[0]

if pred == 1:
    print("\n🚨 FRAUDULENT transaction")
else:
    print("\n✅ Legitimate transaction")
