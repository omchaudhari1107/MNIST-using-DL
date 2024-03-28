## MNIST Handwritten Digit Classification using Deep Learning with accuracy of 97.640001%

### Import's required
```
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
```
### Load the Dataset
```
x,y=keras.datasets.mnist.load_data()
```
- The best accuracy is getting with 28*28=784 neurons as input layer and hidden layer have 100 Neurons and the output layer have 10 Neurons
```
model = keras.Sequential([
    # keras.layer.Dense(Output_shape,input_shape)
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)
```

