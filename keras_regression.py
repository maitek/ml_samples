from __future__ import print_function
import argparse
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import keras.regularizers as regularizers
from dataset import CsvDataset
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

parser = argparse.ArgumentParser(description='Neural network regression sample')
parser.add_argument('csv_file')
args = parser.parse_args()

batch_size = 128
iterations = 20000

train_data = CsvDataset(csv_file=args.csv_file, train=True)
test_data = CsvDataset(csv_file=args.csv_file, train=False)

epochs = int(iterations/len(train_data))+1

x_train = train_data.data
x_test = test_data.data
y_train = train_data.targets
y_test = test_data.targets

# normalize targets
mu = np.mean(y_train)
sigma = np.std(y_train)
y_train = (y_train-mu)/sigma
y_test = (y_test-mu)/sigma

# Plot data
plt.figure()
for i in range(train_data.num_features):
    plt.subplot(1,train_data.num_features,i+1)
    plt.scatter(x_train[:,i],y_train)
    plt.xlabel("x{}".format(i))
    if i == 0:
        plt.ylabel("y")
plt.show()


model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(train_data.num_features,), kernel_regularizer=regularizers.l2(1e-5)))
model.add(Dropout(0.1))
model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
model.add(Dropout(0.1))
model.add(Dense(train_data.num_targets, activation='linear'))

model.summary()

# using mean_absolute_error which is more robust to outliers
model.compile(loss="mean_absolute_error",
              optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5),
              metrics=['mae'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

y_pred = model.predict(x_test)
plt.figure()
plt.scatter(y_pred,y_test)
diag = np.linspace(np.min(y_test),np.max(y_test))
plt.plot(diag,diag, c="red")
plt.xlabel("Predicted value")
plt.ylabel("Ground truth")
plt.show()

print('Test loss:', score[0])
print('Test absolute error:', score[1])
