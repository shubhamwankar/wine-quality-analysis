from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def model_predict(vals_list):
    vals_dict = {"alcohol": vals_list[0],
            "sulphates": vals_list[1],
            "total_sulfur_dioxide": vals_list[2],
            "chlorides": vals_list[3],
            "citric_acid": vals_list[4],
            "volatile_acidity": vals_list[5]}
    input_order = ['volatile_acidity','citric_acid','chlorides',
                    'total_sulfur_dioxide', 'sulphates',
                    'alcohol']
    ordered_vals = [vals_dict[i] for i in input_order]
    vals = np.array(ordered_vals).reshape(1, -1)
    scalar = scalar = joblib.load('scalar.gz')
    scaled_vals = scalar.transform(vals)
    model = pickle.load(open('xgb_model.pkl', 'rb'))
    prediction = model.predict(scaled_vals)
    return prediction[0]


app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        vals_list = request.form.to_dict()
        vals_list = list(vals_list.values())[:-1]
        vals_list = list(map(float, vals_list))
        result = model_predict(vals_list)
        print(result)
        if int(result) == 1:
            prediction = 'Good Quality Wine!'
        else:
            prediction = 'Bad Quality Wine!'
        return render_template('prediction.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
