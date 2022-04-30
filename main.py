import json
import numpy as np
import pickle
from flask import Flask, request, jsonify

# define a flask app
app = Flask(__name__)

model = pickle.load(open('models/final_ai_model.pkl', 'rb'))

@app.route('/status',methods=['POST'])
def status():
    # get the data from the POST request.
    data = request.get_json(force=True)
    # make prediction using model loaded from disk as per the data.
    data_predict = np.array(data['data']).reshape((1, -1))
    prediction = model.predict(data_predict)

    # prediction probability
    prediction_probability = np.max(model.predict_proba(data_predict), axis=1)

    # take the first value of prediction
    final_status = prediction[0]
    return jsonify(status=json.dumps(final_status.astype(float)), probability=json.dumps(prediction_probability[0].astype(float)))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
