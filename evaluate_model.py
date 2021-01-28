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
    y_max = np.amax(predictions)
    y_min = np.amin(predictions)

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
    y_max_naive = np.amax(mean_y)
    y_min_naive = np.amin(mean_y)

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


if __name__ == "__main__":
    # d = prep_data(data_file="data/fiets_1_maart_5_april_uur.csv")

    # df = batchify(d, n_per_group = 24, pckle = False)
    # df.to_pickle("data/windowed_data.pkl")

    # df = pd.read_pickle('data/windowed_data.pkl')
    # print(df.head())
    # train, test = prepare_train_test(df, 5, 0.2)

    # to_pkl = (train, test)
    # pickle.dump(to_pkl, open("testtrain5nodes.p", "wb"))
    # with open("testtrain5nodes.p", "wb") as f:
    #     pickle.dump(to_pkl, f)

    # train, test = pickle.load(open("testtrain5nodes.p", "rb"))
    print("2nodes")
    with open("testtrain2nodes.p", "rb") as f:
        train, test, mean_y = pickle.load(f)

    # Set test and train
    train_x, train_y = train
    test_x, test_y = test

    # Make shapes correct
    # s = train[0].shape
    # train_x = train_x.transpose((0, 3, 1, 2)).reshape((-1, s[3], s[1]*s[2]))
    # train_y = train_y.transpose((0, 2, 1))
    #
    # test_x = test_x.transpose((0, 3, 1, 2)).reshape((-1, s[3], s[1]*s[2]))
    # test_y = test_y.transpose((0, 2, 1))

    model = Model(training_data=(train_x, train_y), validation_data=(test_x, test_y), epochs=100,
                  input_feature_amount=train_x.shape[2],
                  output_feature_amount=train_y.shape[2])

    model.build_model()
    # model.predict(train_x[0], plot=True, y=train_y[0])
    hst = model.train()
    # model.visualize_loss(hst)
    # model.model.load_model("data/checkpoint2021-01-23_211350.h5")
    model.predict(test_x[40], plot=True, y=test_y[40])
    model.predict(test_x[41], plot=True, y=test_y[41])
    model.predict(test_x[42], plot=True, y=test_y[42])
    model.predict(test_x[43], plot=True, y=test_y[43])
    model.predict(test_x[44], plot=True, y=test_y[44])

    calc_rmse(model, test_x, test_y, mean_y, all_samples=True)

    generate_naive_prediction(test_y, mean_y)
