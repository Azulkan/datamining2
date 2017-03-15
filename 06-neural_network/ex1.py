# -*-encoding:utf-8-*

import gzip
import pickle
import numpy as np
import scipy as sp
import keras as kr
import matplotlib as mpl

## Loading data
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

## Separating data and labels from the tuples
x_train = train_set[0]
y_train = train_set[1]

x_valid = valid_set[0]
y_valid = valid_set[1]

x_test = test_set[0]
y_test = test_set[1]

## Converting labels into categorical labels
y_train = kr.utils.np_utils.to_categorical(y_train, 10)
y_test = kr.utils.np_utils.to_categorical(y_test, 10)

## Printing informations on the data
print(y_train)
print(y_valid)
print(y_test)

print("Type de x_train", type(x_train))

print("Forme de x_train : ", x_train.shape)
print("Forme de x_valid : ", x_valid.shape)
print("Forme de x_test : ", x_test.shape)

## Model keras (=Sequential)
model = kr.models.Sequential()

## Layers (=Dense)
model.add(kr.layers.Dense(30, input_dim=784, init='uniform', activation='relu'))
model.add(kr.layers.Dense(10, init='uniform', activation='sigmoid'))

model.compile(loss=kr.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

## Learn the model
# nb_epoch = nb of iterations,
# batch_size = number of instances evaluated before weight update in the network
model.fit(x_train, y_train, nb_epoch=50, batch_size=10)

## Evaluate the model
score = model.evaluate(x_test, y_test)
print("Score with metric %s : %.2f%%" % (model.metrics_names[1], score[1]*100))

## Saving the model
model.save('model_layers-30-10_epoch50.h5')
