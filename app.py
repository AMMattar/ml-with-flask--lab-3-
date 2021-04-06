from flask import Flask, request, render_template
from joblib import load
import pickle # if error

app = Flask(__name__)
model = load('my_ML.joblib') # or load('ml.pkl') if error

@app.route('/', methods=["POST", 'GET'])
def home():
    if request.method == "POST":
        dia = [[float(x) for x in request.form.values()]]
        predict = model.predict(dia)
        return render_template('answer.html', predict=predict)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)