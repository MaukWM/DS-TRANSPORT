# alle nodes pakken voor uur data -> voorspellen voor een node elk uur voor 24 uur
# kijken naar verschil dichtsbijzendste 5 nodes of alle nodes, krijg je beter resultaten? Wat is de impact op performance?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pickle as pkl

pd.set_option('max_columns', None)
pd.set_option('display.max_rows', None)

# metadata = pd.read_csv("data/fietsdata_metadata.csv", nrows=4)
# print(metadata)
#
# fietsdata = pd.read_csv("data/fiets_27_april_31_mei_uur.csv", nrows=24*7)
# print(fietsdata)
#
# fietsdata = fietsdata.loc[fietsdata['naam'] == "Lange Kleiweg (westzijde) tussen Delft en Rijswijk (thv A4)"]
# # print(fietsdata)
#
# fietsdata.drop(fietsdata.columns.difference(['intensiteit_oplopend', 'intensiteit_aflopend', 'intensiteit_beide_richtingen']), 1, inplace=True)
#
# ax = plt.gca()
#
# fietsdata.plot(kind='line', ax=ax) #x='naam'
#
# plt.show()

# We drop these columns from our data as they add nothing.
columns_to_drop = ["type", "naam", "meetperiode", "gebruikte_minuten", "peiling", "betrouwbaarheid", "partnercode",
                   "partner", "aannemer", "meetapparatuur", "licentiecategorie", "beschrijving"]


def clean_data(data_file):
    # Read file
    data = pd.read_csv(data_file)

    # First remove unnecessary columns
    data = data.drop(columns_to_drop, 1)

    # TODO: Check for NaNs in data
    return data


def save_data_to_file(data):
    return


def prep_data(data_file, is_minute_resolution=True):
    # First clean the data
    data = pd.read_csv(data_file)
    data = data.drop(columns_to_drop, 1)

    # print(data["locatiecode"].unique())

    # Drop a broken datapoint
    data = data[data["locatiecode"] != "RDH03_RK03B"]

    # Change timestamp to make data cyclical: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
    data["begintijd"] = pd.to_datetime(data["begintijd"])

    # Extract specific time type
    data['month'] = pd.DatetimeIndex(data['begintijd']).month
    data['day'] = pd.DatetimeIndex(data['begintijd']).day
    data['hour'] = pd.DatetimeIndex(data['begintijd']).hour

    # Convert to be cyclical
    data['month_sin'] = np.sin(2 * np.pi * data["month"] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data["month"] / 12)

    data['day_sin'] = np.sin(2 * np.pi * data["day"] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data["day"] / 7)

    data['hour_sin'] = np.sin(2 * np.pi * data["hour"] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data["hour"] / 24)

    if is_minute_resolution:
        data['minute'] = pd.DatetimeIndex(data['begintijd']).minute

        data['minute_sin'] = np.sin(2 * np.pi * data["minute"] / 60)
        data['minute_cos'] = np.cos(2 * np.pi * data["minute"] / 60)

        data = data.drop("minute", 1)

    # Drop non cyclical columns
    # data = data.drop(["hour", "day", "month", "begintijd"], 1)
    data = data.drop(["hour", "day", "month"], 1)

    # Drop irrelevant part of location id
    data["locatiecode"] = [x.split("_")[1] for x in data["locatiecode"]]

    # Load metadata
    metadata = pd.read_csv("data/fietsdata_metadata.csv")

    # Extract latitude and longitude from single column
    metadata['lat'] = [x.split(",")[0] for x in metadata["Lat/long"]]
    metadata['long'] = [x.split(",")[1] for x in metadata["Lat/long"]]

    # Only keep lat/long columns
    metadata = metadata[["Meetpunt", "lat", "long"]]
    metadata = metadata.rename(columns={"Meetpunt": "locatiecode"})

    # Print for checking unique ids
    # print(data["locatiecode"].unique())
    # print(metadata["locatiecode"].unique())

    # Combine metadata with data
    combined_data = pd.merge(data, metadata, on="locatiecode")

    # Drop location id as it's not necessary anymore
    # combined_data = combined_data.drop("locatiecode", 1)

    # Make everything numeric
    # combined_data = combined_data.apply(pd.to_numeric)
    cols = [i for i in combined_data.columns if i not in ["locatiecode"]]
    for col in cols:
        combined_data[col] = pd.to_numeric(combined_data[col])

    # TODO: Normalize data
    # Make a plot of every column
    # column_names = combined_data.columns
    # for column in column_names:
    #     axs = combined_data[column].hist(bins=100)
    #     axs.set_title(column)
    #     plt.show()

    # Normalize the data between 0 and 1
    normalized_data = combined_data
    # normalized_data = (combined_data-combined_data.min())/(combined_data.max()-combined_data.min())
    for col in cols:
        normalized_data[col] = (normalized_data[col]-normalized_data[col].min())/(normalized_data[col].max()-normalized_data[col].min())

    # TODO: Calc distance between points
    # TODO: Cut into segments of 24 hours (depending on hour/minute data)
    # TODO: split into test/train
    # TODO: create in and output numpy arrays for RNN
    # TODO: save as python pickle
    # TODO: Add variable distance for output node to all other nodes as constant to neural network.
    return normalized_data


columns_to_listify = ['intensiteit_oplopend', 'intensiteit_aflopend', 'intensiteit_beide_richtingen']


def batchify(df, n_per_group=3, pckle=True):
    static_cols = ['locatiecode', 'lat', 'long']
    bounds = df.begintijd.sort_values().unique()
    # bins = list(zip(bounds[:-2], bounds[2:]))

    bins = list(
        zip(*[bounds[i:-n_per_group + i + 1] if -n_per_group + i + 1 < 0 else bounds[i:] for i in range(n_per_group)]))

    def overlapping_bins(x):
        return pd.Series([l for l in bins if l[0] <= x <= l[-1]])

    df = pd.concat([df, df.begintijd.apply(overlapping_bins).stack().reset_index(1, drop=True)], axis=1).rename(
        columns={0: 'times'}).drop('begintijd', axis=1)

    listify = lambda x: x.tolist()
    first = lambda x: x.iloc[0]
    funcdict = {}
    for col in columns_to_listify:
        funcdict[col] = listify
    for col in static_cols:
        funcdict[col] = first

    res_df = None

    for col in columns_to_listify:
        res = df.groupby(['locatiecode', 'times'])[col].apply(listify)
        if res_df is not None:
            res_df[col] = res
        else:
            res_df = res.to_frame()

    for col in static_cols:
        res = df.groupby(['locatiecode', 'times'])[col].apply(first)
        if res_df is not None:
            res_df[col] = res
        else:
            res_df = res.to_frame()

    # res.to_pickle('data/windowed_data.pkl', protocol=4)
    # with open('data/windowed_data.pkl', 'wb') as f:
    #     pickle.dump(res, f, 3)
    if pckle:
        res_df.to_pickle('data/windowed_data.pkl')

    return res_df


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def prepare_train_test(data, n_closest, split):
    def get_train_or_test(data, test_locs, train):
        locations = data[['locatiecode', 'lat', 'long']]
        locations = locations.drop_duplicates()

        if train:
            locations = locations[~locations.locatiecode.isin(test_locs)]

        loc_index = {loc: i for i, loc in enumerate(locations.locatiecode)}
        locations['loc_index'] = [loc_index[loc] for loc in locations.locatiecode]
        lat = {i: float(l) for i, l in zip(locations.loc_index, locations.lat)}
        long = {i: float(l) for i, l in zip(locations.loc_index, locations.long)}

        distances = np.zeros((len(locations), len(locations)))
        for i in range(len(locations)):
            for j in range(len(locations)):
                distances[i, j] = haversine_np(long[i], lat[i], long[j], lat[j])

        #         data['loc_index'] = [loc_index[loc] for loc in data.locatiecode]

        groups = data.groupby('times')
        x = []
        y = []

        if train:
            to_get = range(len(locations))
        else:
            to_get = [loc_index[loc] for loc in test_locs]

        for time, group in groups:
            for i in range(len(locations)):
                # TODO: select column we want to predict
                y.append(group["intensiteit_beide_richtingen"].take([i]))
                closest_indices = np.argsort(distances[i, :])[-(n_closest + 1):-1]
                x_new = []
                for j in closest_indices:
                    group[columns_to_listify].take([j])
                    x_new.append((distances[i, j], group[columns_to_listify].take([j])))
                x.append(x_new)

        res_y = []
        for y_point in y:
            res_y.append(np.stack(y[0].to_numpy()))
        res_y = np.array(res_y)

        res_x = []
        for x_point in x:
            arrs = []
            for dist, data in x_point:
                d = data.to_numpy()
                arrs.append(np.stack([np.array(l) for l in d[0]] + np.array([dist] * len(d[0][0]))))
            res_x.append(np.stack(arrs))
        res_x = np.array(res_x)

        return res_x, res_y

    locs = data[['locatiecode', 'lat', 'long']]
    locs = locs.drop_duplicates()

    test_locs = locs.locatiecode.sample(frac=split).tolist()

    # (x_train, y_train), (x_train, y_train)
    return get_train_or_test(data, test_locs, True), get_train_or_test(data, test_locs, False)

# d = prep_data(data_file="data/fiets_1_maart_5_april_uur.csv")
# df = batchify(d, n_per_group = 24, pckle = False)
# # x, y = prepare_train_test(df, 5, 0.7)
#
# from rnn_model import Model
#
# train, test = prepare_train_test(df, 5, 0.7)
#
# train_x, train_y = train
# test_x, test_y = test
#
# # dummy_train_data = input_data[:80, :], output_data[:80, :]
# # dummy_test_data = input_data[80:, :], output_data[80:, :]
#
# model = Model(training_data=train, validation_data=test, epochs=100)
# model.build_model()
# hst = model.train()
# # model.visualize_loss(hst)
# model.predict(test_x[0], plot=True, y=test_y[0])