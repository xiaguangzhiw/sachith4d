from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense,Conv2D,Activation,Flatten,MaxPooling2D
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import time 

LOG_DIR = f"{int(time.time())}" 

(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

# plt.imshow(x_train[0],cmap = 'gray')
# plt.show()
x_train = x_train.reshape(-1,28,28,1)

x_test = x_test.reshape(-1,28,28,1)



def build_model(hp):
	model = keras.models.Sequential()

	model.add(Conv2D(hp.Int("input_units",min_value=32,max_value=256,step=32), (3, 3), input_shape=x_train.shape[1:]))
	#Dont need to put min_value, max_value... can just put values
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	for i in range(hp.Int("n_layers",1,4)):
		model.add(Conv2D(hp.Int(f"conv_{i}_units",min_value=32,max_value=256,step=32), (3, 3)))
		model.add(Activation('relu'))
	

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(10))
	model.add(Activation("softmax"))

	model.compile(optimizer="adam",
	              loss="sparse_categorical_crossentropy",
	              metrics=["accuracy"])
	return model


tuner = RandomSearch(
	build_model,
	objective = "val_accuracy",
	max_trials = 1,
	executions_per_trial = 1,
	directory = LOG_DIR
	)

# Executions per trial is the number of times we must run through the training data each time

tuner.search(x=x_train,
			 y=y_train,
			 epochs=1,
			 batch_size=64,
			 validation_data=(x_test,y_test))