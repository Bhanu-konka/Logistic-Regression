from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__, template_folder='templates')

# Load and train model
df = pd.read_csv("C:\\Users\\konka\\Desktop\\student_data_extended.csv")
X = df[['InternalMarks', 'Attendance', 'StudyHours']]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "student_model.pkl")
model = joblib.load("student_model.pkl")

# Home page route
@app.route('/')
def home():
    return render_template("index.html")

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        internal_marks = int(data['internal_marks'])
        attendance = int(data['attendance'])
        study_hours = int(data['study_hours'])

        user_input = pd.DataFrame([[internal_marks, attendance, study_hours]],
                                  columns=['InternalMarks', 'Attendance', 'StudyHours'])

        prediction = model.predict(user_input)[0]
        result = "Pass ✅" if prediction == 1 else "Fail ❌"

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
