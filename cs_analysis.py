from ast import literal_eval
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import scipy.stats as stats
import pandas as pd 
import seaborn as sns
from geopy.distance import geodesic as GD
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

# Setup Data
dev_df = pd.read_csv("human_dev_standard.csv")
df = pd.read_csv("cs_mean.csv")

## GENERAL ANALYSIS
"""
#region general_analysis

# Descriptive statistics
print("Describe")
print(df[["type","distance"]].groupby("type").describe())

# Normality check
# It does not come from normal distribution

# Kruskal test - Statistical diference between groups
print("Kruskal")
print(stats.kruskal(df[df['type'] == "company"]['distance'], df[df['type'] == "education"]['distance'], df[df['type'] == "mixed"]['distance']))

# Pearson test and Spearman test- correlation coeficient
print("PEARSON")
unique_majors = df["type"].unique()
df["dist_trunc"] = round(df["distance"], 0)
new_df = df.groupby(["type", "dist_trunc"]).size().reset_index(name="count")
for major in unique_majors:
   print(major + " "+  str(stats.pearsonr(new_df[new_df['type'] == major]['dist_trunc'], new_df[new_df['type'] == major]['count'])))
for major in unique_majors:
   print(major + " "+  str(stats.spearmanr(new_df[new_df['type'] == major]['dist_trunc'], new_df[new_df['type'] == major]['count'])))

#Plot regression
sns.lmplot(
    x="dist_trunc",
    y="count",
    hue="type",
    data=new_df,
    scatter=False,
)
plt.show()

# Test distance by year
means = df.groupby(['year', 'type'])['distance'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="type")
plt.show()

#Probability distribution
for major in unique_majors:
    stats.probplot(df[df['type'] == major]['distance'], dist="norm", plot=plt)
    plt.title("Probability Plot - " +  major)
    plt.show()

#Boxplot
df.boxplot(by ='type', column =['distance'], grid = False)
plt.show()

#Denstity
new_df.groupby('type')['dist_trunc'].plot(kind='kde')
plt.legend(['Company', 'Education', 'Mixed'], title='Relationship')
plt.xlabel('Distance')
plt.show()

#Probabilty
sns.displot(new_df, x="dist_trunc", hue="type", stat="probability", common_norm=False)
plt.xlabel('Distance')
plt.show()

#Histogram
ax = df.plot.hist(column=["distance"], by="type", figsize=(10, 8))
plt.show()

sns.histplot(
    df, x="distance", y="citations",
    bins=30, pthresh=.05, pmax=.9,
)
plt.show()

unique_majors = df["type"].unique()
for major in unique_majors:
   print(major + " "+  str(stats.pearsonr(df[df['type'] == major]['citations'], df[df['type'] == major]['mean_distance'])))
for major in unique_majors:
   print(major + " "+  str(stats.spearmanr(df[df['type'] == major]['citations'], df[df['type'] == major]['mean_distance'])))

print("Describe")
print(df[["type","citations", "international"]].groupby(["type", "international"]).describe())
unique_majors = df["type"].unique()
for major in unique_majors:
    print(major + str(stats.kruskal(df[(df['type'] == major) & (df['international'] == True)]['citations'],df[(df['type'] == major) & (df['international'] == False)]['citations'])))

mixed_df = df[(df['type'] == "mixed")]
sns.lmplot(
    x="distance",
    y="citations",
    hue="international",
    data=mixed_df,
    scatter=False,
)
plt.show()
    
top_df_m = df[(df['type'] == "mixed")].sort_values(by=['citations'], ascending=False).head(1000)
top_df_c = df[(df['type'] == "company")].sort_values(by=['citations'], ascending=False).head(1000)
top_df_e = df[(df['type'] == "education")].sort_values(by=['citations'], ascending=False).head(1000)
cs_df = pd.concat([top_df_m, top_df_e, top_df_c])
print(cs_df[["type","citations", "international", "distance"]].groupby(["type", "international"]).describe())

#endregion
"""

## CONTINENT ANALYSIS
"""
#region continent

df = df[df["type"] == "company"]
new_df = pd.DataFrame({"continent": [], "collaboration": [], "number": int})
collabs = {"NA": [], "EU": [], "AS": [], "OC": [], "SA": [], "AF": []}
for index, work in df.iterrows():
    continent_list= []
    locations = literal_eval(work["location"])
    first_continent = locations[0]["continent"]
    for continent in locations[1:]:
        continent_list.append(continent["continent"])
    for continent in set(continent_list):
        collabs[str(first_continent)] += [continent]

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

new_df.groupby(["continent", "collaboration"]).sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
    layout=(-1, 3),
)
plt.show()

#endregion
"""

## DEVELOPMENT ANALYSIS
"""
#region development

df = df[df["international"] == True]
#df = df[df["no_dev"] == True]
unique_majors = df["type"].unique()
df["dist_trunc"] = round(df["distance"], 0)
for major in unique_majors:
   print(major + " "+  str(stats.pearsonr(df[df['type'] == major]['mean_index'], df[df['type'] == major]['distance'])))
print("------------")
for major in unique_majors:
   print(major + " "+  str(stats.spearmanr(df[df['type'] == major]['mean_index'], df[df['type'] == major]['citations'])))

# Descriptive statistics
print("Describe")
print(df.groupby('no_dev')['distance'].describe())
print(df.groupby('no_dev')['citations'].describe())

# Normality check
# It does not come from normal distribution

# Kruskal test - Statistical diference between groups
print("Kruskal")
print(stats.kruskal(df[df['no_dev'] == True]['distance'], df[df['no_dev'] == False]['distance']))
print(stats.kruskal(df[df['no_dev'] == True]['citations'], df[df['no_dev'] == False]['citations']))

# Pearson test and Spearman test- correlation coeficient
print("CORRELATION TESTS")
df["dist_trunc"] = round(df["distance"], 0)
new_df = df.groupby(["no_dev", "dist_trunc"]).size().reset_index(name="count")
print("Correlation distance and number of papers")
print("No developed" + " "+  str(stats.pearsonr(new_df[new_df['no_dev'] == True]['dist_trunc'], new_df[new_df['no_dev'] == True]['count'])))
print("No developed" + " "+  str(stats.spearmanr(new_df[new_df['no_dev'] == True]['dist_trunc'], new_df[new_df['no_dev'] == True]['count'])))
print("Developed" + " "+  str(stats.pearsonr(new_df[new_df['no_dev'] == False]['dist_trunc'], new_df[new_df['no_dev'] == False]['count'])))
print("Developed" + " "+  str(stats.spearmanr(new_df[new_df['no_dev'] == False]['dist_trunc'], new_df[new_df['no_dev'] == False]['count'])))

print("Correlation HDI and citations")
print("No developed" + " "+  str(stats.pearsonr(df[df['no_dev'] == True]['citations'], df[df['no_dev'] == True]['mean_index'])))
print("No developed" + " "+  str(stats.spearmanr(df[df['no_dev'] == True]['citations'], df[df['no_dev'] == True]['mean_index'])))
print("Developed" + " "+  str(stats.pearsonr(df[df['no_dev'] == False]['citations'], df[df['no_dev'] == False]['mean_index'])))
print("Developed" + " "+  str(stats.spearmanr(df[df['no_dev'] == False]['citations'], df[df['no_dev'] == False]['mean_index'])))

#Index by citations
sns.lmplot(
    x="mean_index",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.show()

#Index by distance
sns.lmplot(
    x="mean_index",
    y="distance",
    hue="type",
    data=df,
    scatter=False,
)
plt.show()

#citations by distance
sns.lmplot(
    x="citations",
    y="distance",
    hue="type",
    data=df,
    scatter=False,
)
plt.show()

# mean distance by development
df.groupby(['no_dev', 'type'])['distance'].mean().unstack().plot(kind='bar')
plt.show()

# mean citations by development
df.groupby(['no_dev', 'type'])['citations'].mean().unstack().plot(kind='bar')
plt.show()

print(df[["type","citations", "no_dev", "distance"]].groupby(["type", "no_dev"]).describe())
no_dev_df = df[["type", "citations", "no_dev", "distance"]]
no_dev_df.groupby(["type", "no_dev"])['citations'].count().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.show()
no_dev_df.groupby(["type", "no_dev"])['citations'].sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.show()

print(stats.kruskal(df[(df['type'] == "mixed") & (df['no_dev'] == False)]['citations'], df[(df['type'] == "mixed") & (df['no_dev'] == True)]['citations']))

top_df_no_dev = df[(df['no_dev'] == True)].sort_values(by=['citations'], ascending=False).head(1000)
top_df_dev = df[(df['no_dev'] == False)].sort_values(by=['citations'], ascending=False).head(1000)
cs_df = pd.concat([top_df_no_dev, top_df_dev])
print(cs_df[["type","citations", "no_dev", "distance"]].groupby(["type", "no_dev"]).describe())

#endregion
"""

## SCOPE ANALYSIS
"""
#region scope

#Descriptive statistics

def f(row):
    if row['ratio'] == 0.5:
        val = "half"
    elif row['ratio'] > 0.5:
        val = "com"
    else:
        val = "edu"
    return val

df = df[df["ratio"] < 1]
df = df[df["ratio"] > 0]

df['ratio_type'] = df.apply(f, axis=1)

print(df[["ratio_type","citations"]].groupby("ratio_type").describe())
print(df[["ratio_type","distance"]].groupby("ratio_type").describe())

print(stats.kruskal(df[df['ratio_type'] == "edu"]['distance'], df[df['ratio_type'] == "com"]['distance'], df[df['ratio_type'] == "half"]['distance']))
print(stats.kruskal(df[df['ratio_type'] == "edu"]['citations'], df[df['ratio_type'] == "com"]['citations'], df[df['ratio_type'] == "half"]['citations']))

sns.lmplot(
    x="ratio",
    y="citations",
    data=df,
    scatter=False,
)
plt.show()

df.groupby(['ratio_type'])['distance'].mean().plot(kind='bar')
plt.show()
df.groupby(['ratio_type'])['citations'].mean().plot(kind='bar')
plt.show()

df.groupby(["ratio_type"])['citations'].mean().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.show()

#endregion
"""
