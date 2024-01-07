import tensorflow as tf
import numpy as np
import pandas
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
import webbrowser
import os


 

file = "Heart_Disease_Prediction.csv"
dataframe = pandas.read_csv(file)

val_dataframe = dataframe.sample(frac=0.2, random_state=2300)
train_dataframe = dataframe.drop(val_dataframe.index)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("Heart Disease")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)


train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


	# Categorical features encoded as integers
age = keras.Input(shape=(1,), name="Age", dtype="int64")
sex = keras.Input(shape=(1,), name="Sex", dtype="int64")
chest_pain_type = keras.Input(shape=(1,), name="Chest pain type", dtype="int64")
bp = keras.Input(shape=(1,), name="BP", dtype="int64")
cholestrol = keras.Input(shape=(1,), name="Cholesterol", dtype="int64")
fbs_over_120 = keras.Input(shape=(1,), name="FBS over 120", dtype="int64")
ekg = keras.Input(shape=(1,), name="EKG results", dtype="int64")
maxHR = keras.Input(shape=(1,), name="Max HR", dtype="int64")
exercise = keras.Input(shape=(1,), name="Exercise angina", dtype="int64")
depress = keras.Input(shape=(1,), name="ST depression", dtype="int64")
slope = keras.Input(shape=(1,), name="Slope of ST", dtype="int64")
vessels = keras.Input(shape=(1,), name="Number of vessels fluro", dtype="int64")
thallium = keras.Input(shape=(1,), name="Thallium", dtype="int64")


all_inputs = [
    age,
    sex,
    chest_pain_type,
    bp,
    cholestrol,
    fbs_over_120,
    ekg,
    maxHR,
    exercise,
    depress,
    slope,
    vessels,
    thallium,
]

# Integer categorical features
age_encoded = encode_categorical_feature(age, "Age", train_ds, False)
sex_encoded = encode_categorical_feature(sex, "Sex", train_ds, False)
pain_encoded = encode_categorical_feature(chest_pain_type, "Chest pain type", train_ds, False)
exercise_encoded = encode_categorical_feature(exercise, "Exercise angina", train_ds, False)

# Numerical features
bp_encoded = encode_numerical_feature(bp, "BP", train_ds)
cholestrol_encoded = encode_numerical_feature(cholestrol, "Cholesterol", train_ds)
fbs_encoded = encode_numerical_feature(fbs_over_120, "FBS over 120", train_ds)
ekg_encoded = encode_numerical_feature(ekg, "EKG results", train_ds)
maxHR_encoded = encode_numerical_feature(maxHR, "Max HR", train_ds)
depress_encoded = encode_numerical_feature(depress, "ST depression", train_ds)
vessels_encoded = encode_numerical_feature(vessels, "Number of vessels fluro", train_ds)
thallium_encoded = encode_numerical_feature(thallium, "Thallium", train_ds)
slope_encoded = encode_numerical_feature(slope, "Slope of ST", train_ds)

all_features = layers.concatenate(
    [
			age_encoded,
			sex_encoded,
			pain_encoded,
			exercise_encoded,

			bp_encoded,
			cholestrol_encoded,
			fbs_encoded,
			ekg_encoded,
			maxHR_encoded,
			depress_encoded,
			vessels_encoded,
			thallium_encoded,
			slope_encoded,
    ]
)
x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

#keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

model.fit(train_ds, epochs=5, validation_data=val_ds)

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# #model.save_weights("model.h5
model.save("heart_disease_model.h5")
