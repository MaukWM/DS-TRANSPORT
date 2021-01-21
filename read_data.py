import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

pd.set_option('max_columns', None)
pd.set_option("max_colwidth", None)

# lines = pd.read_csv("data/intensiteit-snelheid-rotterdam-1-7-Maart/intensiteit-snelheid-rotterdam-1-7-Maart.csv", nrows=20)
# print(lines)
#
#
# metadata = pd.read_csv("data/intensiteit-snelheid-export 1 maart_metadata/intensiteit-snelheid-export_1 maart_metadata.csv", nrows=20)
# print(metadata)

# fietsdata = pd.read_csv("data/fiets_1_maart_5_april_minuut.csv")
# fietsdata1 = pd.read_csv("data/fiets_1_maart_5_april_uur.csv")
# fietsdata2 = pd.read_csv("data/fiets_27_april_31_mei_minuut.csv")
fietsdata3 = pd.read_csv("data/fiets_27_april_31_mei_uur.csv", nrows=24*7)
# print(fietsdata)

fietsdata3 = fietsdata3.loc[fietsdata3['naam'] == "Lange Kleiweg (westzijde) tussen Delft en Rijswijk (thv A4)"]
# print(fietsdata)

# fietsdata.drop(fietsdata.columns.difference(['intensiteit_beide_richtingen']), 1, inplace=True)

ax = plt.gca()
ax1 = plt.gca()
ax2 = plt.gca()
ax3 = plt.gca()

# fietsdata.plot(kind='line', x='naam', ax=ax) #y='intensiteit_beide_richtingen'
# fietsdata1.plot(kind='line', x='naam', ax=ax1) #y='intensiteit_beide_richtingen'
# fietsdata2.plot(kind='line', x='naam', ax=ax2) #y='intensiteit_beide_richtingen'
fietsdata3.plot(kind='line', x='naam', ax=ax3) #y='intensiteit_beide_richtingen'

plt.show()
