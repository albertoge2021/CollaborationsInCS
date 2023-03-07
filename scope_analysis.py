import pandas as pd
import plotly.express as px
import seaborn as sns
from ast import literal_eval
from geopy.distance import geodesic as GD
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import f_oneway
from scipy.stats import shapiro

# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

"""# Setup Data
df = pd.read_csv("cs.csv")
df = df[df["distance"] > 0]
df = df[df["year"] > 1980]
df = df[df["type"] == 'mixed']

for index, work in df.iterrows():
    locations = literal_eval(work["location"])
    education = 0
    company = 0
    for location in locations:
        location = dict(location)
        if location["type"] == "Company":
            company += 1
        elif location["type"] == "Education" :
            education += 1
    if education/(education+company) == 0.5:
        ratio = 'half'
    elif education/(education+company) > 0.5:
        ratio = 'edu'
    if education/(education+company) < 0.5:
        ratio = 'com'
    df._set_value(index,'ratio', ratio)
    df._set_value(index,'ratio_num', education/(education+company))
df.to_csv("cs_scope.csv")"""

df = pd.read_csv("cs_scope.csv")
df = df[df["ratio_num"] < 1]
df = df[df["year"] <= 2021]

"""#Descriptive statistics
print(df[["ratio","citations"]].groupby("ratio").describe())
print(df[["ratio","distance"]].groupby("ratio").describe())

print(stats.kruskal(df[df['ratio'] == "edu"]['distance'], df[df['ratio'] == "com"]['distance'], df[df['ratio'] == "half"]['distance']))
print(stats.kruskal(df[df['ratio'] == "edu"]['citations'], df[df['ratio'] == "com"]['citations'], df[df['ratio'] == "half"]['citations']))

means = df.groupby(['year', 'ratio'])['distance'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="ratio")
plt.show()
means = df.groupby(['year', 'ratio'])['citations'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="ratio")
plt.show()

df["dist_trunc"] = round(df["distance"], 0)
new_df = df.groupby(["ratio", "dist_trunc"]).size().reset_index(name="count")
sns.lmplot(
    x="dist_trunc",
    y="count",
    hue="ratio",
    data=new_df,
    scatter=False,
)
plt.show()"""

sns.lmplot(
    x="ratio_num",
    y="citations",
    data=df,
    scatter=False,
)
plt.show()
