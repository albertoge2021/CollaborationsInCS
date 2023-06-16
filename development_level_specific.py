import pandas as pd
import plotly.express as px
import seaborn as sns
from ast import literal_eval
from geopy.distance import geodesic as GD
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import Counter
import statistics
from scipy.stats import f_oneway
from scipy.stats import shapiro
import pycountry_convert as pc
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

# https://unctad.org/topic/least-developed-countries/list
dev_countries = [
    "AO",
    "BJ",
    "BF",
    "BI",
    "CF",
    "TD",
    "KM",
    "CD",
    "DJ",
    "ER",
    "ET",
    "GM",
    "GN",
    "GW",
    "LS",
    "LR",
    "MG",
    "MW",
    "ML",
    "MR",
    "MZ",
    "NE",
    "RW",
    "ST",
    "SN",
    "SL",
    "SO",
    "SS",
    "SD",
    "TG",
    "UG",
    "TZ",
    "ZM",
    "AF",
    "BD",
    "BT",
    "KH",
    "LA",
    "MM",
    "NP",
    "TL",
    "YE",
    "HT",
    "KI",
    "SB",
    "TV",
]
# Setup Data

"""dev_df = pd.read_csv("human_dev.csv")
for index, row in dev_df.iterrows():
    code = pc.country_alpha3_to_country_alpha2(row["Code"])
    dev_df._set_value(index,'Code_2', code)
dev_df.to_csv("human_dev_standard.csv")"""

"""dev_df = pd.read_csv("human_dev_standard.csv")
df = pd.read_csv("cs_all_test.csv")
df = df.drop_duplicates()
df = df[df["distance"] > 0]
df = df[df["year"] > 1990]
df['no_dev'] = False
df['international'] = False

for index, work in df.iterrows():
    try:
        locations = literal_eval(work["location"])
        location_list = []
        first_country = locations[0]["country"]
        for country in locations[1:]:
            if country["country"] != first_country:
                df._set_value(index,'international', True)
        for location in locations:
            location = dict(location)
            location_list.append(list(dev_df.loc[(dev_df['Year']== work["year"]) & (dev_df["Code_2"] == location["country"])]["Index"])[0])
            if location["country"] in dev_countries:
                df._set_value(index,'no_dev', True)
        df._set_value(index,'mean_index', statistics.mean(location_list))
    except:
        continue
df.to_csv("no_dev.csv")"""


"""
Good!
df = pd.read_csv("no_dev.csv")
df = df[df["international"] == True]
sns.lmplot(
    x="mean_index",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.show()
sns.lmplot(
    x="mean_index",
    y="distance",
    hue="type",
    data=df,
    scatter=False,
)
plt.show()"""

"""df = pd.read_csv("no_dev.csv")
df = df[df["international"] == True]
df = df[df["no_dev"] == True]

print(df.describe())

sns.lineplot(data=df, x="year", y="distance", hue="type")
plt.show()"""
