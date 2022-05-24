# Car price predicion
import pickle
from flask import Flask, request

car_app = Flask(__name__)

def car_predictor(data):
    loaded_model = pickle.load(open("car_pickle.pkl", "rb"))
    result = loaded_model.predict(data)
    return result

@car_app.route("/result", methods=["POST"])
def result():
    if request.method=='POST':
        data = request.json['input']
        print(data)
        prediction = car_predictor(data)
        return f"{prediction}"

if __name__ == "__main__":
    car_app.run(host = "0.0.0.0", port=5000, debug=True)
