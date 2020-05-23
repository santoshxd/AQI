import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import json


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

# @app.route('/index')
# def index():
#     return render_template('index1.html')


@app.route('/Linear')
def Linear():
    return render_template('Linear.html') 

@app.route('/KNN')
def KNN():
    return render_template('KNN.html')

@app.route('/Decision')
def Decision():
    return render_template('Decision.html')

@app.route('/RForest')
def RForest():
    return render_template('RForest.html')

@app.route('/ANN')
def ANN():
    return render_template('ANN.html')

@app.route('/Lasso')
def Lasso():
    return render_template('Lasso.html')


@app.route('/predictLinear',methods=['POST'])
def predictLinear():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('DecisionModel.pkl', 'rb'))
    model1 = pickle.load(open('RFModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    prediction1 = model1.predict(final_features)
    output = abs(round(prediction[0], 2))+0.42
    output1 = abs(round(prediction1[0], 2))+0.17
    output = ((output+output1)/2)+1.69
    if output>=300:
        output=output/3.55;
    output = round(output, 2)
    return render_template('PLinear.html', prediction_text='Air Quality Index (PM 2.5) = {} '.format(output))

@app.route('/predictANN',methods=['POST'])
def predictANN():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('ANNModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('ANN.html', prediction_text='Air Quality Index (PM 2.5) = {}'.format(output))


@app.route('/predictKNN',methods=['POST'])
def predictKNN():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('KNNModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('KNN.html', prediction_text='Air Quality Index (PM 2.5) = {} '.format(output))


@app.route('/predictDecision',methods=['POST'])
def predictDecision():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('DecisionModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = abs(round(prediction[0], 2))

    return render_template('PDecision.html', prediction_text='Air Quality Index (PM 2.5) = {} '.format(output))

@app.route('/predictRF',methods=['POST'])
def predictRF():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('RFModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = abs(round(prediction[0], 2))

    return render_template('PForest.html', prediction_text='Air Quality Index (PM 2.5) = {} '.format(output))

@app.route('/predictLasso',methods=['POST'])
def predictLasso():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('LassoModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('Lasso.html', prediction_text='Air Quality Index (PM 2.5) = {} '.format(output))


if __name__ == "__main__":
    app.run(debug=True)
