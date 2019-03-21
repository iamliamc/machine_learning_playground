import pandas as pd
import numpy as np
import keras

# We define number of examples for training, validation and testing (length*train, length*valid, length*test)
length = 1000
train = 0.8
valid =  0.1
test = 0.1

# We create random features data with five signal (A,B,C,D,E) each signal being 2 days long with a frequency
# of 15min for each example ( or line, tuple or whatever)
df_features = pd.DataFrame(np.random.randn(96*2*length, 5), columns=list('ABCDE'))
df_features["ptu"] = df_features.index % 192
df_features = df_features.set_index([df_features.index // 192, 'ptu'])
df_features = df_features.unstack()

# We create random labels data corresponding to the features (L) being a one day long signal of frequency 15min
# for each example
df_labels = pd.DataFrame(np.random.randn(96*length, 1), columns=list('L'))
df_labels["ptu"] = df_labels.index % 96
df_labels = df_labels.set_index([df_labels.index // 96, 'ptu'])
df_labels = df_labels.unstack()

print(df_features)
print(df_labels)

# We shuffle the data and split them into training, validation and testing dataset
indices = np.random.permutation(length)
x_train = df_features.values[indices[0:int(train * length)]]
y_train = df_labels.values[indices[0:int(train * length)]]
x_valid = df_features.values[indices[int(train * length):int((train + valid) * length)]]
y_valid = df_labels.values[indices[int(train * length):int((train + valid) * length)]]
x_test = df_features.values[indices[int((train + valid) * length):]]
y_test = df_labels.values[indices[int((train + valid) * length):]]

# We create the model
# Input shape MUST be the same shape as the input data.
# Last layer size MUST be the same as the labels shape.
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(960,)))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(96, activation='relu'))

# We compile the model with an optimizer and a loss function
model.compile(optimizer="adam",loss='mean_squared_error')
# We train the model with our training and validation datasets
model.fit(x=x_train,y=y_train,validation_data=(x_valid, y_valid),batch_size=32,epochs=10)
# We make prediction using our testing dataset
y_pred = model.predict(x=x_test)

# Here we go we can compare the prediction against the actual labels
print(y_pred)
print(y_test)