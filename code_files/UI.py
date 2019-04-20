import pickle as pk
import numpy as np

#opening the model
with open("/Users/maor/Documents/src/Iris_flower_classification/model.pickle", "rb") as m:
    model = pk.load(m)

#gtiing the input of the user
inpts = []
categories = ["sepal length(cm)", "sepal width(cm)", "petal length(cm)", "petal width(cm)"]
for category in categories:
    inpts.append(input(f"{category}: "))

#formating the input and predicting it
inpt = np.array(inpts, dtype = np.float64).reshape(1,-1)
prediction = model.predict(inpt)

#printing the prediction to the screen as the flower type
types = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
print(f"Prediction: {types[int(prediction[0])]}")