from ast import literal_eval
from collections import Counter
import pandas as pd
import warnings
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

# Setup Data
df = pd.read_csv("cs_mean.csv")
unique_collaboration_types = df["type"].unique()
## CONTINENT - COUNTRY ANALYSIS

eu_selected_countries = [
    "AT",
    "BE",
    "BG",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GR",
    "HR",
    "HU",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
]
Path("computer_science/continent_analysis/").mkdir(parents=True, exist_ok=True)
Path("computer_science/country_analysis/").mkdir(parents=True, exist_ok=True)

df[["type", "citations", "international"]].groupby(
    ["type", "international"]
).describe().to_csv(
    "computer_science/continent_analysis/describe_citations_by_continent_by_type.csv"
)
with open(
    "computer_science/continent_analysis/kruskal_citations_by_international_by_type.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Kruskal Test for citations by type"
            + str(
                str(
                    stats.kruskal(
                        df[
                            (df["type"] == collaboration_type)
                            & (df["international"] == True)
                        ]["citations"],
                        df[
                            (df["type"] == collaboration_type)
                            & (df["international"] == False)
                        ]["citations"],
                    )
                )
            )
        )

for collaboration_type in unique_collaboration_types:
    collab_df = df[df["type"] == collaboration_type]
    new_df = pd.DataFrame({"continent": [], "collaboration": [], "number": int})
    collabs = {"NA": [], "EU": [], "AS": [], "OC": [], "SA": [], "AF": []}
    for index, work in collab_df.iterrows():
        continent_list = []
        locations = literal_eval(work["location"])
        for continent in locations:
            continent_list.append(continent["continent"])
        for i in range(len(continent_list)):
            for j in range(i + 1, len(continent_list)):
                collabs[continent_list[i]] += [continent_list[j]]
                collabs[continent_list[j]] += [continent_list[i]]
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

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
    new_df.groupby(["continent", "collaboration"]).sum().unstack().plot.pie(
        subplots=True,
        autopct="%1.1f%%",
        legend=False,
        startangle=90,
        figsize=(10, 7),
        layout=(-1, 3),
    )
    fig.suptitle("Collaboration by Continent", fontsize=16, fontweight="bold")
    plt.savefig(
        f"computer_science/continent_analysis/pie_continent_collaboration_by_type_{collaboration_type}.png"
    )
    plt.close()

collabs = {"NA": [], "EU": [], "AS": [], "OC": [], "SA": [], "AF": []}
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    continent_list = []
    for continent in locations:
        continent_list.append(continent["continent"])
    for i in range(len(continent_list)):
        for j in range(i + 1, len(continent_list)):
            collabs[continent_list[i]] += [continent_list[j]]
            collabs[continent_list[j]] += [continent_list[i]]
new_df = pd.DataFrame({"continent": [], "collaboration": [], "number": int})
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

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
new_df.groupby(["continent", "collaboration"]).sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
    layout=(-1, 3),
)
fig.suptitle("Collaboration by Continent", fontsize=16, fontweight="bold")
plt.savefig(f"computer_science/continent_analysis/pie_continent_collaboration.png")
plt.close()