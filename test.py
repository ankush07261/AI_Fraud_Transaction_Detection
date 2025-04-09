import pandas as pd
import joblib

# 1. Load the trained model and scaler
model = joblib.load("model.pkl")     # Load trained IsolationForest model
scaler = joblib.load("scaler.pkl")   # Load saved StandardScaler
print("[INFO] Model and scaler loaded.")

# 2. Load the new test data (make sure format matches training data)
new_data = pd.read_csv("new_test.csv")
print("[INFO] New data loaded. Shape:", new_data.shape)

# 3. Scale the 'Amount' and 'Time' columns using the same scaler
new_data['scaled_amount'] = scaler.transform(new_data['Amount'].values.reshape(-1, 1))
new_data['scaled_time'] = scaler.transform(new_data['Time'].values.reshape(-1, 1))

# 4. Drop original 'Amount' and 'Time' and reorder columns
new_data.drop(['Amount', 'Time'], axis=1, inplace=True)
X_new = new_data[['scaled_amount', 'scaled_time'] + [col for col in new_data.columns if col not in ['scaled_amount', 'scaled_time']]]

# 5. Make predictions with the trained model
raw_preds = model.predict(X_new)

# 6. Map predictions: -1 → Fraud (1), 1 → Not Fraud (0)
new_data['predicted_class'] = [1 if pred == -1 else 0 for pred in raw_preds]
new_data['status'] = new_data['predicted_class'].map({0: "Not Fraud", 1: "Fraud"})

# 7. Display result in a readable format
print("\n--- Prediction Results ---")
for idx, row in new_data.iterrows():
    print(f"Transaction {idx+1}: {row['status']}")

# 8. Optionally, save detailed output
new_data.to_csv("predicted_results.csv", index=False)
print("\n[INFO] Results saved to 'predicted_results.csv'")
