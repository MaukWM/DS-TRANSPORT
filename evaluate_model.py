import math

from prepare_data import prep_data, batchify, prepare_train_test
import pickle
from rnn_model import Model
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt


# Evaluate mean error
def calc_rmse(model, test_x, test_y, mean_y, sample_size=200, all_samples=False):
    # First make all the predictions
    predictions = []
    # print(test_x.shape)

    # Sample random points
    if all_samples:
        sample_size = test_x.shape[0]
        sampled_test_x = test_x
        sampled_test_y = test_y
        mean_y_sampled = mean_y
    else:
        sample_idx = np.random.choice(test_x.shape[0], sample_size)
        sampled_test_x = test_x[sample_idx]
        sampled_test_y = test_y[sample_idx]
        mean_y_sampled = mean_y[sample_idx]



    # Make predictions
    for test_point in sampled_test_x:
        predictions.append(model.predict(test_point))

    predictions = np.stack(predictions)

    # print(np.stack(predictions).shape)
    # print(test_y.shape)

    # Calculate difference
    y_max = np.amax(sampled_test_y)
    y_min = np.amin(sampled_test_y)

    # Root mean squared error
    rmse = math.sqrt(np.sum(np.square(predictions - sampled_test_y)) / sample_size)
    print("RMSE: {0:.2f}".format(rmse))

    # Normalized root mean squared error
    nrmse = rmse / (y_max - y_min) * 100
    print("NRMSE: {0:.2f}".format(nrmse))

    # Mean error
    me = np.sum(predictions - sampled_test_y) / sample_size
    print("ME: {0:.2f}".format(me))

    # Standard deviation
    sd = np.std(predictions - sampled_test_y)
    print("SD: {0:.2f}".format(sd))

    # Naive model predict
    # Calculate difference
    y_max_naive = np.amax(sampled_test_y)
    y_min_naive = np.amin(sampled_test_y)

    # RMSE
    rmse_naive = math.sqrt(np.sum(np.square(mean_y_sampled - sampled_test_y)) / sample_size)
    print("NAIVE RMSE: {0:.2f}".format(rmse_naive))

    # Normalized root mean squared error
    nrmse_naive = rmse_naive / (y_max_naive - y_min_naive) * 100
    print("NAIVE NRMSE: {0:.2f}".format(nrmse_naive))

    # Mean error
    me = np.sum(mean_y_sampled - sampled_test_y) / sample_size
    print("NAIVE ME: {0:.2f}".format(me))

    # Standard deviation
    sd = np.std(mean_y_sampled - sampled_test_y)
    print("NAIVE SD: {0:.2f}".format(sd))


def calculate_accuracy_per_time_step(test_x, test_y, mean_y, model):
    """
    For one model, calculate acc per time step
    """
    # yyy = 2500
    # test_y = test_y[:yyy]
    # mean_y = mean_y[:yyy]
    # test_x = test_x[:yyy]

    naive_model_diffs = abs(test_y - mean_y)
    naive_model_err_per_time_step = np.mean(naive_model_diffs, axis=0)

    plt.grid(color='lightgray', linestyle='--')

    rnn_preds = []
    for i in range(len(test_x)):
        rnn_preds.append(model.predict(test_x[i], plot=False))

    rnn_model_diffs = abs(test_y - rnn_preds)
    rnn_model_err_per_time_step = np.mean(rnn_model_diffs, axis=0)

    # plt.plot(np.reshape(test_y[0], newshape=(24 ,1)), label="actual0")
    # plt.plot(np.reshape(mean_y[0], newshape=(24 ,1)), label="naive0")
    # plt.plot(np.reshape(rnn_preds[0], newshape=(24 ,1)), label="rnn0")

    plt.plot(naive_model_err_per_time_step, label="naive_model")
    plt.plot(rnn_model_err_per_time_step, label="rnn_model")
    plt.title("Average error per time step (8 nodes)")

    plt.legend()
    plt.ylim([0, None])

    plt.show()


def calculate_accuracy_per_time_step_all_models(models_dict):
    """
    For all models, calculate acc per time step
    """

    plt.grid(color='lightgray', linestyle='--')

    for key in models_dict.keys():
        print("Processing key: " + str(key))

        model, test_x, test_y, mean_y = models_dict[key]

        # 8,9 coloumn for time sin 0 = 0, cos 0 = 1

        # Heel smerig dit
        indices = [i for i, x in enumerate(test_x) if x[0][7] == 0.5 and x[0][8] == 1]

        # print(test_x[0][0])
        # print(test_x[0][1])

        print(len(indices))

        test_x = np.stack([test_x[i] for i in indices])
        test_y = np.stack([test_y[i] for i in indices])
        mean_y = np.stack([mean_y[i] for i in indices])

        # print(test_x.shape)
        # print(test_y.shape)

        # Uncomment this to cut out a portion of the samples
        # yyy = 500
        # test_y = test_y[:yyy]
        # mean_y = mean_y[:yyy]
        # test_x = test_x[:yyy]

        naive_model_diffs = abs(test_y - mean_y)
        naive_model_err_per_time_step = np.mean(naive_model_diffs, axis=0)

        rnn_preds = []
        for i in range(len(test_x)):
            rnn_preds.append(model.predict(test_x[i], plot=False))

        rnn_model_diffs = abs(test_y - rnn_preds)
        rnn_model_err_per_time_step = np.mean(rnn_model_diffs, axis=0)

        plt.plot(naive_model_err_per_time_step, label="naive_model" + str(key))
        plt.plot(rnn_model_err_per_time_step, label="rnn_model" + str(key))

    plt.title("Average error per time step")

    plt.xlabel("Timestep (24 hours)")
    plt.ylabel("Mean error")

    plt.legend()
    plt.ylim([0, None])

    plt.show()


# Make prediction with naive model
def generate_naive_prediction(test_y, mean_y, plot=True):
    sample_idx = np.random.choice(test_y.shape[0], 1)

    y_real = test_y[sample_idx]
    y_naive_pred = mean_y[sample_idx]

    if plot:
        # print(y.shape, result.shape)
        # print("y", y)
        plt.plot(y_real[0], label="real")

        # print("result", result)
        plt.plot(y_naive_pred[0], label="predicted")

        plt.title(label="Prediction")
        plt.legend()

        plt.ylabel("Intensity (both directions)")
        plt.xlabel("Timestep (1 hour)")

        plt.show()


def plot_loss_from_hist(hist_dict):
    plt.plot(hist_dict.history['loss'], label="loss")
    plt.plot(hist_dict.history['val_loss'], label="val_loss")

    plt.legend()

    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    plt.show()


if __name__ == "__main__":
    print(8)
    with open("testtrain8.p", "rb") as f:
        train_x, train_y, test_x8, test_y8, mean_y8 = pickle.load(f)

    model8 = Model(training_data=(train_x, train_y), validation_data=(test_x8, test_y8), epochs=400,
                   input_feature_amount=train_x.shape[2],
                   output_feature_amount=train_y.shape[2])

    model8.build_model()
    model8.model.load_weights("models/8nodesmodel.h5")

    print(5)
    with open("testtrain5.p", "rb") as f:
        train_x, train_y, test_x5, test_y5, mean_y5 = pickle.load(f)

    model5 = Model(training_data=(train_x, train_y), validation_data=(test_x5, test_y5), epochs=400,
                   input_feature_amount=train_x.shape[2],
                   output_feature_amount=train_y.shape[2])

    model5.build_model()
    model5.model.load_weights("models/5nodesmodel.h5")

    print(2)
    with open("testtrain2.p", "rb") as f:
        train_x, train_y, test_x2, test_y2, mean_y2 = pickle.load(f)

    model2 = Model(training_data=(train_x, train_y), validation_data=(test_x2, test_y2), epochs=400,
                   input_feature_amount=train_x.shape[2],
                   output_feature_amount=train_y.shape[2])

    model2.build_model()
    model2.model.load_weights("models/2nodesmodel.h5")

    models_dict = {'2': (model2, test_x2, test_y2, mean_y2), '5': (model5, test_x5, test_y5, mean_y5),
                   '8': (model8, test_x8, test_y8, mean_y8)}

    calculate_accuracy_per_time_step_all_models(models_dict)

    # model.predict(train_x[0], plot=True, y=train_y[0])
    # hst = model.train()
    # model.visualize_loss(hst)
    # model.model.load_model("data/checkpoint2021-01-23_211350.h5")
    # model.predict(test_x[40], plot=True, y=test_y[40])
    # model.predict(test_x[41], plot=True, y=test_y[41])
    # model.predict(test_x[42], plot=True, y=test_y[42])
    # model.predict(test_x[43], plot=True, y=test_y[43])
    # model.predict(test_x[44], plot=True, y=test_y[44])

    # calc_rmse(model, test_x, test_y, mean_y, all_samples=True)

    # generate_naive_prediction(test_y, mean_y)

    # calculate_accuracy_per_time_step(test_x, test_y, mean_y)

    # plot_loss_from_hist(hst)
