import pandas as pd
import plotly.express as px
import seaborn as sns
from ast import literal_eval
from geopy.distance import geodesic as GD
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import Counter
from scipy.stats import f_oneway
from scipy.stats import shapiro
import pycountry_convert as pc
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

"""# Setup Data
df = pd.read_csv("cs_all_test.csv")
df = df.drop_duplicates()
df = df[df["distance"] > 0]
df = df[df["year"] > 1980]

for index, work in df.iterrows():
    locations = literal_eval(work["location"])
    new_loc = []
    for location in locations:
        location = dict(location)
        location["continent"] = pc.convert_country_alpha2_to_continent_code.country_alpha2_to_continent_code(location["country"])
        new_loc.append(location)
    df._set_value(index,'location',new_loc)
df.to_csv("cs_continents.csv")"""

df = pd.read_csv("cs_continents.csv")

new_df = pd.DataFrame({"continent": str, "collaboration": str, "type":str, "citations": int, "year":int, "concepts":[]})
for index, work in df.iterrows():
    locations = literal_eval(work["location"])
    first_continent = locations[0]["continent"]
    for continent in locations[1:]:
        new_df = new_df.append(
            pd.Series(
                [first_continent, continent["continent"], work["type"], work["citations"], work["year"], work["concepts"]],
                index=new_df.columns,
            ),
            ignore_index=True,
        )
new_df.to_csv("continent_analysis.csv")
