from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pickle
import traceback
import pandas as pd
import numpy as np



app = Flask(__name__)

@app.route('/')
def home():
    return render template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	lr = joblib.load("reg.pkl") # Load "model.pkl"

	model_columns = joblib.load("model_reg.pkl") # Load "model_columns.pkl"

    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = lr.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Solar Production should be kW {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)


