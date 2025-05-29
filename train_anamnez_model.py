import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


df = pd.read_csv("anamnez_data.csv")

df["gender"] = LabelEncoder().fit_transform(df["gender"])

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

X = df.drop(["label", "label_encoded"], axis=1)
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

model.save_model("anamnez_model.json")
print("✅ Model başarıyla kaydedildi → anamnez_model.json")

import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
