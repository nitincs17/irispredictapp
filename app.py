import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('irispredict.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sepLength = float(request.form['sepLength'])
    sepWidth = float(request.form['sepWidth'])
    petLength = float(request.form['petLength'])
    petWidth = float(request.form['petWidth'])
    
    finalFeatures = np.array([[sepLength,sepWidth,petLength,petWidth]])
    prediction = model.predict(finalFeatures)

    

    return render_template('index.html', prediction_text='Predicted Species is  $ {}'.format(round(prediction[0][0])))


if __name__ == "__main__":
    app.run(debug=True)