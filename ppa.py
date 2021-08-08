# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 11:55:25 2021

@author: Odd
"""
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.tree import DecisionTreeClassifier




app = Flask(__name__)
model = pickle.load(open('model.plk', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST', 'GET'])
def predict(): 
    intf = [int(x) for x in request.form.values()]
    ff   = [np.array(intf)]
    pred = model.predict(ff)
    output = round(pred[0, 2]
    return 
    











if __name__ == '__main__':
    app.run(debug=True)

     