from prepare_data import prep_data, batchify, prepare_train_test
from rnn_model import Model
import numpy as np
import pickle
import pandas as pd


def prep_and_save_data(neighbours_amount):
    generate_data = True
    pkl_file = "testtrain" + str(neighbours_amount) + ".p"
    if generate_data:
        data_files = ["data/fiets_27_april_31_mei_uur.csv", "data/fiets_1_maart_5_april_uur.csv"]
        time_units = 24
        n_neighbours = neighbours_amount
        results = []

        for file in data_files:
            d = prep_data(data_file=file)
            df = batchify(d, n_per_group = time_units, pckle = False)
            df.to_pickle("data/windowed_data.pkl")

            # df = pd.read_pickle('data/windowed_data.pkl')

            result = prepare_train_test(df, n_neighbours, 0.2)
            results.append(result)

        to_pkl = tuple([np.concatenate(list(t)) for t in zip(*results)])

        train_x, train_y, test_x, test_y, means_y = to_pkl
        pickle.dump(to_pkl, open(pkl_file, "wb"))
        with open(pkl_file, "wb") as f:
            pickle.dump(to_pkl, f)
    else:
        with open(pkl_file, "rb") as f:
            train_x, train_y, test_x, test_y, means_y = pickle.load(f)


prep_and_save_data(8)

# print(4)
#
#
# print(train_x.shape, train_y.shape)
# print(test_x.shape, test_y.shape)
# print(means_y.shape)
# print(5)
#
# model = Model(training_data=(train_x, train_y), validation_data=(test_x, test_y), epochs=100, input_feature_amount=train_x.shape[2],
#               output_feature_amount=train_y.shape[2])
#
# print(6)
# model.build_model()
# # model.predict(train_x[0], plot=True, y=train_y[0])
# hst = model.train()
# # model.visualize_loss(hst)
# # model.model.load_model("data/checkpoint2021-01-23_211350.h5")
# model.predict(test_x[40], plot=True, y=test_y[40])
# model.predict(test_x[41], plot=True, y=test_y[41])
# model.predict(test_x[42], plot=True, y=test_y[42])
# model.predict(test_x[43], plot=True, y=test_y[43])
# model.predict(test_x[44], plot=True, y=test_y[44])
#
# # Evaluate the model
# loss, acc = model.model.evaluate(test_x, test_y, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
