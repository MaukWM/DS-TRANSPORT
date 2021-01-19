# alle nodes pakken voor uur data -> voorspellen voor een node elk uur voor 24 uur
# kijken naar verschil dichtsbijzendste 5 nodes of alle nodes, krijg je beter resultaten? Wat is de impact op performance?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

pd.set_option('max_columns', None)

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
# TODO: Verify minute and hour data contain exactly the same columns
# TODO: Verify whether type is always the same and what the difference might be
columns_to_drop = ["type", "naam", "meetperiode", "gebruikte_minuten", "peiling", "betrouwbaarheid", "partnercode",
                   "partner", "aannemer", "meetapparatuur", "licentiecategorie", "beschrijving"]


def clean_data(data_file):
    # Read file
    data = pd.read_csv(data_file)

    # First remove unnecessary columns
    data = data.drop(columns_to_drop, 1)

    # TODO: Check for NaNs in data
    return data


def prep_hour_data(data_file):
    # First clean the data
    data = pd.read_csv(data_file)
    data = data.drop(columns_to_drop, 1)

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

    # Drop non cyclical columns
    data = data.drop(["hour", "day", "month", "begintijd"], 1)

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
    combined_data = combined_data.drop("locatiecode", 1)

    # print(combined_data.head())


    # TODO: Normalize data
    # TODO: Cut into segments of 24 hours (depending on hour/minute data)
    # TODO: split into test/train
    # TODO: create in and output numpy arrays for RNN
    # TODO: save as python pickle
    # TODO: Add variable distance for output node to all other nodes as constant to neural network.
    return


prep_hour_data("data/fiets_27_april_31_mei_uur.csv")


