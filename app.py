import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('mdly.pkl',  'rb'))



@app.route('/')
def home():
    return render_template('indexe.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('indexe.html', prediction_text='PROBABILITY THAT YOUR LOAN WILL GET APPROVED IS ; {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)