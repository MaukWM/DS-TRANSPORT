import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":

    df = pd.read_csv("data/fiets_1_maart_5_april_uur.csv")

    df['hour'] = pd.DatetimeIndex(df['begintijd']).hour

    df = df[['hour', 'intensiteit_beide_richtingen']]

    df = df.groupby('hour').mean()

    plt.grid(color='lightgray', linestyle='--')

    plt.plot(df['intensiteit_beide_richtingen'], label='average intensity')
    plt.xlabel('intensity')
    plt.ylabel('timestep (24 hours)')

    plt.title('Average intensity (both directions) per timestep')

    plt.legend()

    plt.show()

    # with open("testtrain2.p", "rb") as f:
    #     train_x, train_y, test_x, test_y, mean_y = pickle.load(f)
    #
    # cut_train = int(len(train_y)/24)
    # cut_test = int(len(test_y)/24)
    # train_y = train_y[:cut_train]
    # test_y = test_y[:cut_test]
    #
    # # train_y = np.reshape(train_y, newshape=(cut, 24))
    # # test_y = np.reshape(test_y, newshape=(cut, 24))
    #
    # print(train_y.shape)
    # print(test_y.shape)
    #
    # ys = np.append(train_y, test_y, axis=0)
    #
    # print(ys.shape)
    #
    # avg_per_timestep = np.mean(ys, axis=0)
    #
    # print(avg_per_timestep.shape)
    #
    # plt.plot(avg_per_timestep)
    # plt.show()
