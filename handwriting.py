import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers.experimental import preprocessing

sns.set(style="whitegrid")
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# Using the MNIST hadnwriting dataset from keras
mnist = keras.datasets.mnist

# Split the training and validation datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding the labels
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Reshape the dataset to (28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Create a convolutional neural network
model = keras.Sequential([
    # Data Augmentation
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomFlip('vertical'),
    preprocessing.RandomContrast(0.5),
    preprocessing.RandomWidth(factor=0.15),
    preprocessing.RandomRotation(factor=0.20),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),

    # Input layer
    layers.Input((28, 28, 1)), 
    
    # Feature extraction

    # First convolutional block, with 64 filters, each with a 3*3 kernal
    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding="same"), 
    layers.MaxPool2D(pool_size=2), 

     # Second Convolutional Block
    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    # Third Convolutional Block
    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.Flatten(), 

    # Add a hidden layer, randomly dropout some nodes to prevent overfitting
    layers.Dense(128, activation="relu"), 
    layers.Dropout(0.5), 
    layers.BatchNormalization(), 

    # Add another hidden layer
    layers.Dense(256, activation="relu"), 
    layers.Dropout(0.5), 
    layers.BatchNormalization(), 

    # output layer
    layers.Dense(10, activation="softmax")
])

# Train the neural network
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test),
                    epochs=10, 
                    batch_size=32, 
                    verbose=2)

model.evaluate(x_test, y_test, verbose=2)

# Visualization of the result
import pandas as pd

# show loss
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# show accuracy
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


# Save the trained model to filename
filename = "model.keras"
if len(sys.argv) == 2:
    filename = sys.argv[1]
model.save(filename)
print(f"Trainning Process Completed! Model saved to {filename}.")
