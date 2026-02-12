# -----------------------------
# Interactive Crop Prediction (Multiple Predictions)
# -----------------------------

import pandas as pd
import pickle

# Step 1: Load trained model and label encoder
with open(r"D:\models CR\crop_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(r"D:\models CR\label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Step 2: Function to get validated input
def get_input(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(f"{prompt} [{min_val}-{max_val}]: "))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"⚠ Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("⚠ Invalid input! Enter a valid number.")

print("\n🌱 Welcome to Crop Recommendation System!\n")

# Step 3: Loop for unlimited predictions without restart
while True:
    print("\nEnter Environment Parameters:\n")

    nitrogen = get_input("Nitrogen (N)", 50, 140)
    phosphorus = get_input("Phosphorus (P)", 30, 60)
    potassium = get_input("Potassium (K)", 40, 45)
    temperature = get_input("Temperature (°C)", 15, 35)
    humidity = get_input("Humidity (%)", 30, 90)
    ph = get_input("pH Value", 5.5, 7.5)
    rainfall = get_input("Rainfall (mm)", 50, 300)

    # Prepare input for model
    user_df = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
                           columns=['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall'])

    # Prediction
    predicted_crop = le.inverse_transform(model.predict(user_df))
    print(f"\n🌾 Recommended Crop: {predicted_crop[0]}\n")

    # Continue or exit
    again = input("Do you want to predict again? (yes/no): ").strip().lower()
    if again != "yes":
        print("\n✨ Thank you for using the Crop Recommendation System!")
        break
