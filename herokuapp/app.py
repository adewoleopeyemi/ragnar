# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:51:03 2020

@author: adewole opeyemi
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0])
    return render_template('index.html', prediction_text='prediction is' + str(output))

if __name__ == '__main__':
    app.run(port=5002)
    
    