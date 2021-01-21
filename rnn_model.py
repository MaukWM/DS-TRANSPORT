import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras import layers

import matplotlib.pyplot as plt

# model = ks.Sequential()
# # Add an Embedding layer expecting input vocab of size 1000, and
# # output embedding dimension of size 64.
# model.add(layers.Embedding(input_dim=1000, output_dim=64))
#
# # Add a LSTM layer with 128 internal units.
# model.add(layers.LSTM(128))
#
# # Add a Dense layer with 10 units.
# model.add(layers.Dense(10))
#
# model.summary()


class Model:

    def __init__(self, data=None, batch_size=256, state_size=10, input_feature_amount=10, output_feature_amount=1,
                 seq_len_in=24, seq_len_out=24, steps_per_epoch=100, epochs=20, learning_rate=0.00075):

        self.data = data
        self.batch_size = batch_size
        self.state_size = state_size
        self.input_feature_amount = input_feature_amount
        self.output_feature_amount = output_feature_amount
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out

        # TODO: set data to generator

        # Create input dummy set, with 100 datapoints, of 24 hours, with 50 input features
        self.input_data = np.zeros((100, 24, 50)) + 1
        # Create output dummy set, with 100 datapoints, of 24 hours, with 1 output feature
        self.output_data = np.zeros((100, 24, 1)) + 2

        self.test_train_ratio = 0.5

        # TODO: Split dynamically based on length data
        self.train_x, self.test_x = self.input_data[:80, :], self.input_data[80:, :]
        self.train_y, self.test_y = self.output_data[:80, :], self.output_data[80:, :]

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.learning_rate = learning_rate

        # To be determined
        self.model = None
        self.plot_loss = False

    def build_model(self):

        inputs = ks.layers.Input(shape=(self.input_data.shape[1], self.input_data.shape[2]))
        lstm_out = ks.layers.LSTM(self.state_size)(inputs)
        outputs = ks.layers.Dense(1)(lstm_out)

        self.model = ks.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=ks.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        self.model.summary()

    def generate_validation_data(self):
        test_x_batch = []
        test_y_batch = []

    def train(self):
        es_callback = ks.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

        modelckpt_callback = ks.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath="models/checkpoint" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".h5",
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
        )

        callbacks = [es_callback, modelckpt_callback]

        # TODO: Fit on generator

        val = (self.test_x, self.test_y)
        history = self.model.fit(x=self.train_x, y=self.train_y, steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs, validation_data=val, callbacks=callbacks)

        return history

    def visualize_loss(self, history):
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title("Loss visualisation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


model = Model()
model.build_model()
hst = model.train()
model.visualize_loss(hst)
