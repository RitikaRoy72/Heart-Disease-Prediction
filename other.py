import tensorflow as tf
import numpy as np
import pandas
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, redirect

from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
import main

model = load_model("hear_disease_model.h5")

form = main.FieldStorage()
age = form.getValue("age")
sex = form.getValue("sex")
chest_pain_type = form.getValue("pain")
bp = form.getValue("bp")
cholesterol = form.getValue("chol")
fbs_over_120 = form.getValue("fbs")
ekg = form.getValue("ekg")
maxHR = form.getValue("hr")
exercise = form.getValue("exercisea")
depress = form.getValue("stdepress")
slope = form.getValue("slope")
vessels = form.getValue("vessels")
thallium = form.getValue("thallium")

sample = {
    "age": age,
    "sex": sex,
    "Chest pain type": chest_pain_type,
    "BP": bp,
    "Cholesterol": cholesterol,
    "FBS over 120": fbs_over_120,
    "EKG results": ekg,
    "Max HR": maxHR,
    "Exercise Angina": exercise,
    "ST depression": depress,
    "Slope of ST": slope,
    "Number of vessels fluro": vessels,
    "Thallium": thallium,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)
f = open("temp.txt", "w")
f.write(predictions)
f.close()

 
 
app = Flask(__name__)
 
 
@app.route("/index.html")
@app.route("/")
def go_to_external_url():
    return redirect('http://google.com')
 
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000)