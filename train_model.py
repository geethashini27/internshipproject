import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv("loan_data.csv")

X = data[['Income', 'LoanAmount', 'CreditScore']]
y = data['Risk']

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')

print("Model trained and saved successfully!")