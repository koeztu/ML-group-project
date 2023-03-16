import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt



#READ DATA AND SELECT FEATURES
data = pd.read_csv('./completeDataset.csv', header=0, index_col=0)


X = data.iloc[:, :-2].to_numpy()
y = data.iloc[:, -1].to_numpy()

print(X)
print(y)
print(X.shape)
print(y.shape)

#create validation data
#shuffle images first
permutation = np.random.permutation(X.shape[0])
X_shuffled = X[permutation]
y_shuffled = y[permutation]

#split data next
#use 50% for training, 25% of data for validation, and 25% for testing
num_train_samples = int(0.7 * X.shape[0])
num_val_samples = int(0.10 * X.shape[0])
X_train = X_shuffled[:num_train_samples]
y_train = y_shuffled[:num_train_samples]
X_val = X_shuffled[num_train_samples:num_train_samples + num_val_samples]
y_val = y_shuffled[num_train_samples:num_train_samples + num_val_samples]
X_test = X_shuffled[num_train_samples + num_val_samples:]
y_test = y_shuffled[num_train_samples + num_val_samples:]

print(X_val.shape)
print(y_val.shape)


#specify model
model = keras.Sequential([
    layers.Dense(40), #dense means fully connected
    layers.LeakyReLU(),
    layers.Dense(40),
    layers.LeakyReLU(),
    layers.Dense(40),
    layers.LeakyReLU(),
    layers.Dropout(0.5),
    layers.Dense(5, activation="softmax")]) #softmax output layer

#compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005), #specify backpropagation algorithm
              loss="sparse_categorical_crossentropy", #specify loss function, spares because we only provide the correct answer (1 or 0) in the labels, not the whole vector.
              metrics=["accuracy"]) #specify metrics you are interested in



#estimate model
history = model.fit(X_train, y_train,
                    epochs=40,
                    batch_size=1,
                    validation_data=(X_val, y_val))

model.summary()

#draw some graphs with matplotlib
history_dict = history.history #keras (model.fit) automatically generates and returns a dictionary with metrics
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
epochs = range(1, len(loss) + 1)

fig = plt.figure(figsize=(18,25)) #figsize sets the size in centimeters
gs = fig.add_gridspec(1,15) #set up grid, which can be later divided into subplots if needed
ax1 = fig.add_subplot(gs[0:1, 0:7]) #first chart
ax2 = fig.add_subplot(gs[0:1, 8:15]) #second chart

ax1.plot(epochs, loss, "b", label="Training loss")
ax1.plot(epochs, val_loss, "r", label="Validation loss")
ax2.plot(epochs, acc, "b", label="Training accuracy")
ax2.plot(epochs, val_acc, "r", label="Validation accuracy")
plt.title("Training and validation loss")
ax1.set_xlabel("Epochs")
ax2.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Accuracy")
ax1.legend()
ax2.legend()
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"test_loss: {test_loss}")
print(f"test_acc: {test_acc}")
