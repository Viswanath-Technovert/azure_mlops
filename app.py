from flask import Flask, render_template, request
import pandas as pd
import pickle


app = Flask(__name__)

# Load label encoders and the best model
with open('best_model.pkl', 'rb') as f:
    best_model, label_encoders = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features from the form
        CreditScore = int(request.form['CreditScore'])
        Geography = request.form['Geography']
        Gender = request.form['Gender']
        Age = int(request.form['Age'])
        Tenure = int(request.form['Tenure'])
        Balance = float(request.form['Balance'])
        NumOfProducts = int(request.form['NumOfProducts'])
        HasCrCard = int(request.form['HasCrCard'])
        IsActiveMember = int(request.form['IsActiveMember'])
        EstimatedSalary = float(request.form['EstimatedSalary'])

        # Convert categorical variables using label encoders
        Geography = label_encoders['Geography'].transform([Geography])[0]
        Gender = label_encoders['Gender'].transform([Gender])[0]

        # Create DataFrame with user input
        data = {
            'CreditScore': [CreditScore],
            'Geography': [Geography],
            'Gender': [Gender],
            'Age': [Age],
            'Tenure': [Tenure],
            'Balance': [Balance],
            'NumOfProducts': [NumOfProducts],
            'HasCrCard': [HasCrCard],
            'IsActiveMember': [IsActiveMember],
            'EstimatedSalary': [EstimatedSalary]
        }

        df = pd.DataFrame(data)

        # Make prediction
        prediction = best_model.predict(df)[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)

