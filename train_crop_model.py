# -----------------------------
# Train & Save Crop Recommendation Model
# -----------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load dataset
data = pd.read_csv(r"D:\models CR\Crop_recommendation.csv")
print("📊 Dataset Loaded Successfully!")
print(data.head())

# Step 2: Prepare features and labels
X = data[['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Step 3: Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"\n🔹 Training Size: {X_train.shape}")
print(f"🔹 Testing Size: {X_test.shape}")

# Step 5: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy*100:.2f}%")
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 7: Save model and label encoder as .pkl files
with open(r"D:\crop recommendation\crop_recommendation_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(r"D:\crop recommendation\label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n💾 Model and Label Encoder saved successfully!")
