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
df = pd.read_csv("cs_all_pre.csv")
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

df = df[df["type"] == "education"]
new_df = pd.DataFrame({"continent": [], "collaboration": [], "number": int})
collabs = {"NA": [], "EU": [], "AS": [], "OC": [], "SA": [], "AF": []}
for index, work in df.iterrows():
    locations = literal_eval(work["location"])
    first_continent = locations[0]["continent"]
    for continent in locations[1:]:
        collabs[str(first_continent)] += [continent["continent"]]
for k, v in collabs.items():
    values = Counter(v)
    for key, value in values.items():
        new_df = new_df.append(
            pd.Series(
                [k, key, value],
                index=new_df.columns,
            ),
            ignore_index=True,
        )

# print(new_df)
# new_df.groupby(['continent','collaboration']).sum().unstack().plot(kind='bar',y='number')
# plt.show()

"""new_df.groupby(["continent", "collaboration"]).sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
    layout=(-1, 3),
)
plt.show()"""

#https://ourworldindata.org/human-development-index

new_df['perc']= 100 * new_df['number'] / new_df.groupby('continent')['number'].transform('sum')
#new_df.groupby(['continent','collaboration']).sum().unstack().plot(kind='bar',y='perc', stacked=True)
#plt.show()
dev_df = pd.read_csv("human_dev.csv")
dev_df = dev_df[dev_df["Year"] == 2021]

for index, work in dev_df.iterrows():
    continent = pc.convert_country_alpha2_to_continent_code.country_alpha2_to_continent_code(pc.country_alpha3_to_country_alpha2(work["Code"]))
    percentage = new_df[(new_df["continent"] == continent) & (new_df["collaboration"] == continent)].iloc[0]['perc']
    dev_df._set_value(index,'continent',continent)
    dev_df._set_value(index,'self_collab', percentage)


means = dev_df.groupby(['continent']).mean("Index")
sns.lineplot(data=means, x="Index", y="self_collab")
plt.show()

sns.lmplot(
    x="Index",
    y="self_collab",
    data=means,
    
)
plt.show()