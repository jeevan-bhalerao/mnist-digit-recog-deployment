from keras.datasets import mnist
# import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)

# single_image = x_train[0]

# print(single_image.shape)

y_example = to_categorical(y_train)

y_cat_test = to_categorical(y_test,10)# last parameter is classes i.e 10 classes

y_cat_train = to_categorical(y_train,10)

# here normalize between 0 to 1
x_train = x_train/255
x_test = x_test/255

# Reshape
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)

model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
# 2d to 1D
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train,y_cat_train,epochs=2)

# Evaluate model 

model.metrics_names

model.evaluate(x_test,y_cat_test)

predictions = model.predict_classes(x_test)

print(classification_report(y_test,predictions))

# save model
model.save('MNIST_model.h5')

# Saving the model for Future Inferences

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")

# from tensorflow.keras.models import save_model
# save_model(model, "MNIST_model.h5")