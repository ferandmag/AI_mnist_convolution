import numpy as np
from flask import Flask, render_template, request
from imageio import imwrite, imread
from skimage.transform import resize
import re
import sys
import os
from load import *

app = Flask(__name__)
global model, graph
# model, graph = init()

@app.route('/')
def index():
    return render_template('index.html')

def convertImage(imgData):
    imstr = re.search(r'base64,(.*'.imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(imstr.decode('base64'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    x = imread('out.png', mode='L')
    x = np.invert(x)
    x = resize(x, 28, 28)
    x = x.reshape(1, 28, 28, 1)
    with graph.as_default():
        out = model.predict(x)
        response = np.array_str(np.argmax(out))
        return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)