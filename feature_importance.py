import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

model = joblib.load("model.pkl")

data = load_breast_cancer()

importance = model.feature_importances_

df = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": importance
})

df = df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(df["Feature"], df["Importance"])
plt.title("Feature Importance")
plt.tight_layout()
plt.show()