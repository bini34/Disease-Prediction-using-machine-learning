from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and prepare the data
def prepare_data():
    data = pd.read_csv('data/Testing.csv')
    X = data.drop(['prognosis'], axis=1)
    y = data['prognosis']
    
    # Get symptoms list and encode disease labels
    symptoms = X.columns.tolist()
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train the model
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y_encoded)
    
    return symptoms, model, le

# Initialize data
symptoms_list, model, label_encoder = prepare_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_symptoms = []
    
    if request.method == 'POST':
        # Get selected symptoms
        selected_symptoms = request.form.getlist('symptoms')
        
        # Create input data array
        input_data = np.zeros(len(symptoms_list))
        for symptom in selected_symptoms:
            symptom_index = symptoms_list.index(symptom)
            input_data[symptom_index] = 1
            
        # Make prediction
        prediction = model.predict([input_data])
        disease = label_encoder.inverse_transform(prediction)[0]
        
        # Get probability scores
        probabilities = model.predict_proba([input_data])[0]
        max_prob = max(probabilities) * 100
        
        return render_template('index.html', 
                             symptoms=symptoms_list,
                             prediction=disease,
                             probability=f"{max_prob:.1f}",
                             selected_symptoms=selected_symptoms)
    
    return render_template('index.html', 
                         symptoms=symptoms_list,
                         prediction=None,
                         selected_symptoms=[])

if __name__ == '__main__':
    app.run(debug=True)
