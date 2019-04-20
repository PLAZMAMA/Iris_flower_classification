import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection, neighbors
import pickle as pk

def to_categorial(arr):
    """converts the classes of the flowers from 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica' to 0,1,2"""
    types = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    categorical = []
    for elem in arr:
        for i in range(len(types)):
            if elem == types[i]:
                categorical.append(i)
    return(categorical)

#loading the dataset
df = pd.read_csv("/Users/maor/Documents/src/Iris_flower_classification/iris.data")

#assinging the xs and ys
xs = np.array(df.drop('class', axis = 1), dtype = np.float64)
ys = np.array(df[['class']])
ys = np.array(to_categorial(ys), dtype = np.float64)

#creating the training and testing dataset
x_train, x_test, y_train, y_test = model_selection.train_test_split(xs, ys, test_size = 0.3)

#defining and training the model
model = neighbors.KNeighborsClassifier()
model.fit(x_train, y_train)

#testing the "accuracy of the model" and printing it out
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy}")

#saving the model
with open("/Users/maor/Documents/src/Iris_flower_classification/model.pickle", "wb") as m:
    pk.dump(model, m)