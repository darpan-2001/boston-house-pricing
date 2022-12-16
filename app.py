import pickle
from flask import Flask, app, request, jsonify,url_for, render_template,redirect, flash, session, escape

import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return 'hello model'

@app.route('/predict', methods=['POST'])
def predict_output():
    data = request.json['data']
    # print(data)
    # print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    # print(output)
    return jsonify(output[0])


if __name__=='__main__':
    app.run(debug=True)