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

    def __init__(self, training_data=None, validation_data=None, batch_size=256, state_size=32, input_feature_amount=10, output_feature_amount=1,
                 seq_len_in=24, seq_len_out=24, steps_per_epoch=100, epochs=20, learning_rate=0.000075):

        self.training_data = training_data
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.state_size = state_size
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out

        self.input_feature_amount = input_feature_amount
        self.output_feature_amount = output_feature_amount

        self.test_train_ratio = 0.5

        self.train_x, self.train_y = training_data
        self.test_x, self.test_y = validation_data
        self.input_shape = (self.train_x.shape[1], self.train_x.shape[2])

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.learning_rate = learning_rate

        # To be determined
        self.model = None
        self.plot_loss = False

    def build_model(self):
        inputs = ks.layers.Input(shape=self.input_shape)
        lstm_out = ks.layers.LSTM(self.state_size, return_sequences=True)(inputs)
        outputs = ks.layers.Dense(self.output_feature_amount)(lstm_out)

        self.model = ks.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=ks.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        self.model.summary()

    def train(self):
        es_callback = ks.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

        modelckpt_callback = ks.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath="models/checkpoint" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".h5",
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
        )

        callbacks = [es_callback, modelckpt_callback]  # TODO: Add back early stopping: es_callback

        val = (self.test_x, self.test_y)
        history = self.model.fit(x=self.train_x, y=self.train_y, steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs, validation_data=val, callbacks=callbacks)

        return history

    def predict(self, x, plot=False, y=None):
        # TODO: Denormalize prediction
        # Expand dimension so that keras doesn't complain
        x = np.expand_dims(x, axis=0)
        # print(self.input_shape, x.shape)
        result = self.model.predict(x)[0]

        if plot:
            # print(y.shape, result.shape)
            # print("y", y)
            plt.plot(y, label="real")

            # print("result", result)
            plt.plot(result, label="predicted")

            plt.title(label="Prediction")
            plt.legend()

            plt.show()

        return result

    @staticmethod
    def visualize_loss(history):
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


# # Create input dummy set, with 100 datapoints, of 24 hours, with 50 input features
# input_data = np.zeros((100, 24, 50)) + 1
# # Create output dummy set, with 100 datapoints, of 24 hours, with 1 output feature
# output_data = np.zeros((100, 24, 1)) + 2
#
# dummy_train_data = input_data[:80, :], output_data[:80, :]
# dummy_test_data = input_data[80:, :], output_data[80:, :]
#
# model = Model(training_data=dummy_train_data, validation_data=dummy_test_data, epochs=100)
# model.build_model()
# hst = model.train()
# # model.visualize_loss(hst)
# model.predict(dummy_train_data[0][0], plot=True, y=dummy_train_data[1][0])
