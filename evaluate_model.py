import math

from prepare_data import prep_data, batchify, prepare_train_test
from rnn_model import Model
import pandas as pd
import pickle
import random
import numpy as np

# d = prep_data(data_file="data/fiets_1_maart_5_april_uur.csv")

# df = batchify(d, n_per_group = 24, pckle = False)
# df.to_pickle("data/windowed_data.pkl")

# df = pd.read_pickle('data/windowed_data.pkl')
# print(df.head())
# train, test = prepare_train_test(df, 8, 0.2)

# to_pkl = (train, test)
# pickle.dump(to_pkl, open("testtrain.p", "wb"))
# with open("testtrain.p", "wb") as f:
#     pickle.dump(to_pkl, f)

# train, test = pickle.load(open("testtrain.p", "rb"))
with open("testtrain.p", "rb") as f:
    train, test = pickle.load(f)

# Set test and train
train_x, train_y = train
test_x, test_y = test

# Make shapes correct
s = train[0].shape
train_x = train_x.transpose((0, 3, 1, 2)).reshape((-1, s[3], s[1]*s[2]))
train_y = train_y.transpose((0, 2, 1))

test_x = test_x.transpose((0, 3, 1, 2)).reshape((-1, s[3], s[1]*s[2]))
test_y = test_y.transpose((0, 2, 1))

model = Model(training_data=(train_x, train_y), validation_data=(test_x, test_y), epochs=100, input_feature_amount=train_x.shape[2],
              output_feature_amount=train_y.shape[2])

model.build_model()
# model.predict(train_x[0], plot=True, y=train_y[0])
# hst = model.train()
# model.visualize_loss(hst)
# model.model.load_model("data/checkpoint2021-01-23_211350.h5")
model.predict(test_x[40], plot=True, y=test_y[40])
model.predict(test_x[41], plot=True, y=test_y[41])
model.predict(test_x[42], plot=True, y=test_y[42])
model.predict(test_x[43], plot=True, y=test_y[43])
model.predict(test_x[44], plot=True, y=test_y[44])


# Evaluate mean error
def calc_rmse(model, test_x, test_y, sample_size=200):
    # First make all the predictions
    predictions = []
    # print(test_x.shape)

    # Sample random points
    sample_idx = np.random.choice(test_x.shape[0], sample_size)

    sampled_test_x = test_x[sample_idx]
    sampled_test_y = test_y[sample_idx]

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


calc_rmse(model, test_x, test_y)


