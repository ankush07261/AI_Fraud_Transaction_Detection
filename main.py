import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib  # Used to save the model as .pkl

# 1. Load the dataset
df = pd.read_csv("creditcard.csv")  # Load the training dataset
print("Data loaded. Shape:", df.shape)

# 2. Scale 'Amount' and 'Time' columns (they’re not already scaled)
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop original columns and rearrange
df.drop(['Amount', 'Time'], axis=1, inplace=True)
df = df[['scaled_amount', 'scaled_time'] + [col for col in df.columns if col not in ['scaled_amount', 'scaled_time', 'Class']] + ['Class']]

# 3. Define features (X) and labels (y)
X = df.drop('Class', axis=1)  # Input features
y = df['Class']               # Labels (not used in training but useful for evaluation)

# 4. Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Train Isolation Forest (Unsupervised anomaly detection model)
model = IsolationForest(
    n_estimators=100,     # Number of trees (random split trees)
    contamination=0.001,  # Estimated % of outliers (fraud)
    random_state=42       # For reproducibility
)
model.fit(X_train)  # Fit model to training data

# 6. Save trained model to .pkl file
joblib.dump(model, "model.pkl")  # Saves model to disk
print("Model trained and saved to model.pkl")

# 7. Save scaler too (for consistent test input scaling)
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved to scaler.pkl")
