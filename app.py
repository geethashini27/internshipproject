from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    income = float(request.form['income'])
    loan_amount = float(request.form['loan_amount'])
    credit_score = float(request.form['credit_score'])

    prediction = model.predict([[income, loan_amount, credit_score]])

    if prediction[0] == 0:
        result = "Safe Customer"
    else:
        result = "Risky Customer"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)