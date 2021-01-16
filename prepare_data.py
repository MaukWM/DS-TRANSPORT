# alle nodes pakken voor uur data -> voorspellen voor een node elk uur voor 24 uur
# kijken naar verschil dichtsbijzendste 5 nodes of alle nodes, krijg je beter resultaten? Wat is de impact op performance?

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# pd.set_option('max_columns', None)
# pd.set_option("max_colwidth", None)

metadata = pd.read_csv("data/fietsdata_metadata.csv", nrows=4)

fietsdata = pd.read_csv("data/fiets_27_april_31_mei_uur.csv", nrows=24*7)
print(fietsdata)

fietsdata = fietsdata.loc[fietsdata['naam'] == "Lange Kleiweg (westzijde) tussen Delft en Rijswijk (thv A4)"]
# print(fietsdata)

fietsdata.drop(fietsdata.columns.difference(['intensiteit_oplopend', 'intensiteit_aflopend', 'intensiteit_beide_richtingen']), 1, inplace=True)

ax = plt.gca()

fietsdata.plot(kind='line', ax=ax) #x='naam'

plt.show()

# We drop these columns from our data as they add nothing.
# TODO: Verify minute and hour data contain exactly the same columns
# TODO: Verify whether type is always the same and what the difference might be
columns_to_drop = ["type", "naam", "meetperiode", "gebruikte_minuten", "peiling", "betrouwbaarheid", "partnercode",
                   "partner", "aannemer", "meetapparatuur", "licentiecategorie", "beschrijving"]

def clean_data(data_file):
    # Read file
    data = pd.read_csv("data/" + data_file)


    # First remove unnecessary columns
    data.drop(columns_to_drop, 1, inplace=True)

    # TODO: Check for NaNs in data

def prep_data(data_file):
    # TODO: replace location_code with long and lat
    # TODO: Change timestamp with sin for hour of day and day of week

    # TODO: Normalize data
    # TODO: Cut into segments of 24 hours (depending on hour/minute data)
    # TODO: split into test/train
    # TODO: create in and output numpy arrays for RNN
    # TODO: save as python pickle
    # TODO: Add variable distance for output node to all other nodes as constant to neural network.
    return



