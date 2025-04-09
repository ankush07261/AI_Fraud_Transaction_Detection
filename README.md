# 🧠 Credit Card Fraud Detection using Isolation Forest

This project uses an **unsupervised anomaly detection** technique to detect frauds in credit card transactions using the Isolation Forest algorithm.

## 🧩 Model Used
- Isolation Forest (tree-based anomaly detection)

## ⚙️ Features
- Preprocessing with StandardScaler
- Class imbalance handling
- Confusion matrix + Classification report

## 🪜 Steps to run:
### Clone the repo:
```sh
git clone https://github.com/ankush07261/AI_Fraud_Transaction_Detection
```
### Install the dependenies in the project dir:
```sh
cd AI_Fraud_Transaction_Detection
pip install -r requirements.txt
```
### Run the main.py
```sh
python main.py
```
### Now, the model is trained and model.pkl is generated to run tests on any other data. Try testing on the new_test.csv by running test.py:
```sh
python test.py
```
