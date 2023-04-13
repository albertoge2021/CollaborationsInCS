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
dev_df = pd.read_csv("human_dev_standard.csv")
df = pd.read_csv("cs_mean.csv")

## GENERAL ANALYSIS


# Descriptive statistics
Path("computer_science/general_analysis/").mkdir(parents=True, exist_ok=True)
df[["type", "distance"]].groupby("type").describe().to_csv(
    "computer_science/general_analysis/describe_max_distance_by_type.csv"
)
df[["type", "mean_distance"]].groupby("type").describe().to_csv(
    "computer_science/general_analysis/describe_mean_distance_by_type.csv"
)

# Normality check
# It does not come from normal distribution

# Kruskal test - Statistical diference between groups
with open(
    "computer_science/general_analysis/kruskal_max_distance_by_type.txt", "w"
) as f:
    f.write(
        "Kruskal Test for Max distance by type"
        + str(
            stats.kruskal(
                df[df["type"] == "company"]["distance"],
                df[df["type"] == "education"]["distance"],
                df[df["type"] == "mixed"]["distance"],
            )
        )
    )
with open(
    "computer_science/general_analysis/kruskal_mean_distance_by_type.txt", "w"
) as f:
    f.write(
        "Kruskal Test for Mean distance by type"
        + str(
            stats.kruskal(
                df[df["type"] == "company"]["mean_distance"],
                df[df["type"] == "education"]["mean_distance"],
                df[df["type"] == "mixed"]["mean_distance"],
            )
        )
    )
with open("computer_science/general_analysis/kruskal_citations_by_type.txt", "w") as f:
    f.write(
        "Kruskal Test for Mean distance by type"
        + str(
            stats.kruskal(
                df[df["type"] == "company"]["citations"],
                df[df["type"] == "education"]["citations"],
                df[df["type"] == "mixed"]["citations"],
            )
        )
    )

# Pearson test and Spearman test- correlation coeficient
unique_collaboration_types = df["type"].unique()
df["dist_trunc"] = round(df["distance"], 0)
max_df = df.groupby(["type", "dist_trunc"]).size().reset_index(name="count")
with open(
    "computer_science/general_analysis/correlation_max_distance_compared_to_count_by_type.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Pearson test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    max_df[max_df["type"] == collaboration_type]["dist_trunc"],
                    max_df[max_df["type"] == collaboration_type]["count"],
                )
            )
        )
        f.write("\n")
        f.write(
            "Spearman test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    max_df[max_df["type"] == collaboration_type]["dist_trunc"],
                    max_df[max_df["type"] == collaboration_type]["count"],
                )
            )
        )
        f.write("\n")

df["mean_dist_trunc"] = round(df["mean_distance"], 0)
mean_df = df.groupby(["type", "mean_dist_trunc"]).size().reset_index(name="count")
with open(
    "computer_science/general_analysis/correlation_mean_distance_compared_to_count_by_type.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Pearson test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    mean_df[mean_df["type"] == collaboration_type]["mean_dist_trunc"],
                    mean_df[mean_df["type"] == collaboration_type]["count"],
                )
            )
        )
        f.write("\n")
        f.write(
            "Spearman test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    mean_df[mean_df["type"] == collaboration_type]["mean_dist_trunc"],
                    mean_df[mean_df["type"] == collaboration_type]["count"],
                )
            )
        )
        f.write("\n")

with open(
    "computer_science/general_analysis/correlation_max_distance_compared_to_citations_by_type.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Pearson test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    df[df["type"] == collaboration_type]["citations"],
                    df[df["type"] == collaboration_type]["distance"],
                )
            )
        )
        f.write("\n")
        f.write(
            "Spearman test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    df[df["type"] == collaboration_type]["citations"],
                    df[df["type"] == collaboration_type]["distance"],
                )
            )
        )
        f.write("\n")

with open(
    "computer_science/general_analysis/correlation_mean_distance_compared_to_citations_by_type.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Pearson test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    df[df["type"] == collaboration_type]["citations"],
                    df[df["type"] == collaboration_type]["mean_distance"],
                )
            )
        )
        f.write("\n")
        f.write(
            "Spearman test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    df[df["type"] == collaboration_type]["citations"],
                    df[df["type"] == collaboration_type]["mean_distance"],
                )
            )
        )
        f.write("\n")
# Plot regression for truncation distance vs count
sns.lmplot(
    x="dist_trunc",
    y="count",
    hue="type",
    data=max_df,
    scatter=False,
)
plt.xlabel("Truncation distance")
plt.ylabel("Count")
plt.title("Scatterplot of maximum truncation distance vs count")
plt.savefig("computer_science/general_analysis/scatter_max_trunc_distance_count.png")
plt.close()

# Plot regression for mean truncation distance vs count
sns.lmplot(
    x="mean_dist_trunc",
    y="count",
    hue="type",
    data=mean_df,
    scatter=False,
)
plt.xlabel("Mean truncation distance")
plt.ylabel("Count")
plt.title("Scatterplot of mean truncation distance vs count")
plt.savefig("computer_science/general_analysis/scatter_mean_trunc_distance_count.png")
plt.close()

# Line plot of mean distance by year by type
means = df.groupby(["year", "type"])["distance"].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="type")
plt.xlabel("Year")
plt.ylabel("Mean distance")
plt.title("Line plot of maximum distance by year by type")
plt.savefig(
    "computer_science/general_analysis/lineplot_max_distance_by_year_by_type.png"
)
plt.close()

# Line plot of mean distance by year by type
means = df.groupby(["year", "type"])["mean_distance"].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="type")
plt.xlabel("Year")
plt.ylabel("Mean distance")
plt.title("Line plot of mean distance by year by type")
plt.savefig(
    "computer_science/general_analysis/lineplot_mean_distance_by_year_by_type.png"
)
plt.close()

# Line plot of mean citations by year by type
means = df.groupby(["year", "type"])["citations"].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="type")
plt.xlabel("Year")
plt.ylabel("Mean citations")
plt.title("Line plot of mean citations by year by type")
plt.savefig(
    "computer_science/general_analysis/lineplot_mean_citations_by_year_by_type.png"
)
plt.close()


"""#Probability distribution
for collaboration_type in unique_collaboration_types:
    stats.probplot(df[df['type'] == collaboration_type]['distance'], dist="norm", plot=plt)
    plt.title("Probability Plot - " +  collaboration_type)
    plt.show()
"""

# Boxplot of distance by type
df.boxplot(by="type", column=["distance"], grid=False)
plt.title("Boxplot of max distance by type")
plt.xlabel("Type")
plt.ylabel("Max distance")
plt.savefig("computer_science/general_analysis/boxplot_max_distance_by_type.png")
plt.close()

df.boxplot(by="type", column=["mean_distance"], grid=False)
plt.title("Boxplot of mean distance by type")
plt.xlabel("Type")
plt.ylabel("Mean distance")
plt.savefig("computer_science/general_analysis/boxplot_mean_distance_by_type.png")
plt.close()

# Density plot of truncation distance by type
max_df.groupby("type")["dist_trunc"].plot(kind="kde")
plt.legend(["Company", "Education", "Mixed"], title="Relationship")
plt.title("Density plot of max truncation distance by type")
plt.xlabel("Max truncation distance")
plt.savefig("computer_science/general_analysis/density_max_trunc_distance_by_type.png")
plt.close()

mean_df.groupby("type")["mean_dist_trunc"].plot(kind="kde")
plt.legend(["Company", "Education", "Mixed"], title="Relationship")
plt.title("Density plot of mean truncation distance by type")
plt.xlabel("Mean truncation distance")
plt.savefig("computer_science/general_analysis/density_mean_trunc_distance_by_type.png")
plt.close()

# Probability plot of truncation distance by type
sns.displot(max_df, x="dist_trunc", hue="type", stat="probability", common_norm=False)
plt.title("Probability plot of max truncation distance by type")
plt.xlabel("Max truncation distance")
plt.savefig(
    "computer_science/general_analysis/probability_max_trunc_distance_by_type.png"
)
plt.close()

sns.displot(
    mean_df, x="mean_dist_trunc", hue="type", stat="probability", common_norm=False
)
plt.title("Probability plot of mean truncation distance by type")
plt.xlabel("Mean truncation distance")
plt.savefig(
    "computer_science/general_analysis/probability_mean_trunc_distance_by_type.png"
)
plt.close()

# Histogram
ax = df.plot.hist(
    column=["distance"],
    by="type",
    figsize=(10, 8),
    xlabel="Distance (meters)",
    ylabel="Frequency",
)
plt.savefig("computer_science/general_analysis/histogram_max_distance_by_type.png")
plt.close()

ax = df.plot.hist(
    column=["mean_distance"],
    by="type",
    figsize=(10, 8),
    xlabel="Mean Distance (meters)",
    ylabel="Frequency",
)
plt.savefig("computer_science/general_analysis/histogram_mean_distance_by_type.png")
plt.close()

ax = sns.histplot(
    df,
    x="distance",
    y="citations",
    bins=30,
    pthresh=0.05,
    pmax=0.9,
)
ax.set(xlabel="Distance (meters)", ylabel="Number of Citations")
plt.savefig(
    "computer_science/general_analysis/histplot_max_distance_compared_to_citations_by_type.png"
)
plt.close()

ax = sns.histplot(
    df,
    x="mean_distance",
    y="citations",
    bins=30,
    pthresh=0.05,
    pmax=0.9,
)
ax.set(xlabel="Mean Distance (meters)", ylabel="Number of Citations")
plt.savefig(
    "computer_science/general_analysis/histplot_mean_distance_compared_to_citations_by_type.png"
)
plt.close()

ax = sns.lmplot(
    x="distance",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
ax.set(xlabel="Mean Distance (meters)", ylabel="Number of Citations")
plt.savefig(
    "computer_science/general_analysis/scatter_max_distance_compared_to_citations_by_type.png"
)
plt.close()

ax = sns.lmplot(
    x="mean_distance",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
ax.set(xlabel="Mean Distance (meters)", ylabel="Number of Citations")
plt.savefig(
    "computer_science/general_analysis/scatter_mean_distance_compared_to_citations_by_type.png"
)
plt.close()


## CONTINENT - COUNTRY ANALYSIS

# region continent
eu_countries = [
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
    continent_df = df[df["type"] == collaboration_type]
    new_df = pd.DataFrame({"continent": [], "collaboration": [], "number": int})
    collabs = {"NA": [], "EU": [], "AS": [], "OC": [], "SA": [], "AF": []}
    for index, work in continent_df.iterrows():
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

for collaboration_type in unique_collaboration_types:
    df = pd.read_csv("cs_mean.csv")
    collabs = {"NA": [], "EU": [], "AS": [], "OC": [], "SA": [], "AF": []}
    df = df[df["type"] == collaboration_type]
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
    plt.savefig(
        f"computer_science/continent_analysis/pie_test_continent_collaboration_by_type_{collaboration_type}.png"
    )
    plt.close()

df = pd.read_csv("cs_eu.csv")
for collaboration_type in unique_collaboration_types:
    df = pd.read_csv("cs_eu.csv")
    new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
    collabs = {"EU": [], "US": [], "CN": []}
    df = df[df["type"] == collaboration_type]
    for row in tqdm(df.itertuples()):
        locations = literal_eval(row.location)
        country_list = []
        for location in locations:
            country_list.append(location["country"])
        for i in range(len(country_list)):
            for j in range(i + 1, len(country_list)):
                if (country_list[i] in collabs.keys()) and (
                    country_list[j] in collabs.keys()
                ):
                    collabs[country_list[i]] += [country_list[j]]
                    collabs[country_list[j]] += [country_list[i]]
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
    new_df.groupby(["country", "collaboration"]).sum().unstack().plot.pie(
        subplots=True,
        autopct="%1.1f%%",
        legend=False,
        startangle=90,
        figsize=(10, 7),
        layout=(-1, 3),
    )
    fig.suptitle("Collaboration by Country", fontsize=16, fontweight="bold")
    plt.savefig(
        f"computer_science/country_analysis/pie_country_collaboration_by_type_{collaboration_type}.png"
    )
    plt.close()

countries = ["EU", "US", "CN"]
for collaboration_type in unique_collaboration_types:
    df = pd.read_csv("cs_eu.csv")
    new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
    collabs = {country: [] for country in countries}
    df = df[df["type"] == collaboration_type]
    for row in tqdm(df.itertuples()):
        locations = literal_eval(row.location)
        country_list = []
        for location in locations:
            country_code = location["country"]
            if country_code in countries:
                country_list.append(country_code)
        for i in range(len(country_list)):
            for j in range(i + 1, len(country_list)):
                collabs[country_list[i]] += [country_list[j]]
                collabs[country_list[j]] += [country_list[i]]
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
    new_df.groupby(["country", "collaboration"]).sum().unstack().plot(
        kind="bar", y="number"
    )
    plt.savefig(
        f"computer_science/country_analysis/bar_country_collaboration_by_type_{collaboration_type}.png"
    )
    plt.close()

countries = ["EU", "US", "CN"]

for collaboration_type in unique_collaboration_types:
    us_collaborations = 0
    eu_collaborations = 0
    cn_collaborations = 0
    us_eu_collaborations = 0
    us_cn_collaborations = 0
    eu_cn_collaborations = 0
    eu_cn_us_collaborations = 0
    df = pd.read_csv("cs_eu.csv")
    df = df[df["type"] == collaboration_type]
    for row in tqdm(df.itertuples()):
        locations = literal_eval(row.location)
        country_list = []
        for location in locations:
            country_code = location["country"]
            if country_code in countries:
                country_list.append(country_code)
        if "US" in set(country_list):
            us_collaborations += 1
            if "CN" in country_list and "EU" in country_list:
                eu_cn_us_collaborations += 1
            if "CN" in country_list:
                us_cn_collaborations += 1
            if "EU" in country_list:
                us_eu_collaborations += 1
        if "CN" in country_list:
            cn_collaborations += 1
            if "EU" in country_list:
                eu_cn_collaborations += 1
        if "EU" in country_list:
            eu_collaborations += 1
    with open(
        f"computer_science/country_analysis/country_collaboration_cn_us_eu_percentage_by_type_{collaboration_type}.txt",
        "w",
    ) as file:
        file.write(f"{collaboration_type}\n")
        file.write(
            f"CN - US collaboration represents {(us_cn_collaborations / us_collaborations) * 100:.2f}% of total US collaborations\n"
        )
        file.write(
            f"CN - US collaboration represents {(us_cn_collaborations / cn_collaborations) * 100:.2f}% of total Chinese collaborations\n"
        )
        file.write(
            f"CN - EU collaboration represents {(eu_cn_collaborations / cn_collaborations) * 100:.2f}% of total CN collaborations\n"
        )
        file.write(
            f"CN - EU collaboration represents {(eu_cn_collaborations / eu_collaborations) * 100:.2f}% of total EU collaborations\n"
        )
        file.write(
            f"EU - US collaboration represents {(us_eu_collaborations / us_collaborations) * 100:.2f}% of total US collaborations\n"
        )
        file.write(
            f"EU - US collaboration represents {(us_eu_collaborations / eu_collaborations) * 100:.2f}% of total EU collaborations\n"
        )

    with open(
        f"computer_science/country_analysis/country_collaboration_cn_us_eu_by_type_{collaboration_type}.txt",
        "w",
    ) as file:
        file.write(f"{collaboration_type}\n")
        file.write(f"CN - US collaboration {(us_cn_collaborations)}\n")
        file.write(f"CN - EU collaboration {(eu_cn_collaborations)}\n")
        file.write(f"EU - US collaboration {(us_eu_collaborations)}\n")
        file.write(f"EU - US - CN collaboration {(eu_cn_us_collaborations)}\n")

    # Define the data
    us_data = [us_collaborations, us_eu_collaborations, us_cn_collaborations]
    eu_data = [us_eu_collaborations, eu_collaborations, eu_cn_collaborations]
    cn_data = [eu_cn_collaborations, us_cn_collaborations, cn_collaborations]

    # Define the x-axis labels
    labels = ["US Collaborations", "EU Collaborations", "CN Collaborations"]

    # Define the x-axis locations for each group of bars
    x_us = [0, 4, 8]
    x_eu = [1, 5, 9]
    x_cn = [2, 6, 10]

    # Plot the bars
    plt.bar(x_us, us_data, color="blue", width=0.8, label="US")
    plt.bar(x_eu, eu_data, color="red", width=0.8, label="EU")
    plt.bar(x_cn, cn_data, color="green", width=0.8, label="CN")

    # Add the x-axis labels and tick marks
    plt.xticks([1.5, 5.5, 9.5], labels)
    plt.xlabel("Collaboration Type")
    plt.ylabel("Number of Collaborations")

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(
        f"computer_science/country_analysis/bar_country_collaboration_cn_us_eu_by_type_{collaboration_type}.png"
    )
    plt.close()

    # Define the data
    us_data = [
        (us_collaborations / us_collaborations) * 100,
        (us_eu_collaborations / us_collaborations) * 100,
        (us_cn_collaborations / us_collaborations) * 100,
    ]
    eu_data = [
        (us_eu_collaborations / eu_collaborations) * 100,
        (eu_collaborations / eu_collaborations) * 100,
        (us_cn_collaborations / eu_collaborations) * 100,
    ]
    cn_data = [
        (us_cn_collaborations / cn_collaborations) * 100,
        (eu_cn_collaborations / cn_collaborations) * 100,
        (cn_collaborations / cn_collaborations) * 100,
    ]

    # Define the x-axis labels
    labels = ["US Collaborations", "EU Collaborations", "CN Collaborations"]

    # Define the x-axis locations for each group of bars
    x_us = [0, 4, 8]
    x_eu = [1, 5, 9]
    x_cn = [2, 6, 10]

    # Plot the bars
    plt.bar(x_us, us_data, color="blue", width=0.8, label="US")
    plt.bar(x_eu, eu_data, color="red", width=0.8, label="EU")
    plt.bar(x_cn, cn_data, color="green", width=0.8, label="CN")

    # Add the x-axis labels and tick marks
    plt.xticks([1.5, 5.5, 9.5], labels)
    plt.xlabel("Collaboration Type")
    plt.ylabel("Number of Collaborations")

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(
        f"computer_science/country_analysis/bar_country_collaboration_cn_us_eu_percent_by_type_{collaboration_type}.png"
    )
    plt.close()

# Define the countries of interest
df = pd.read_csv("cs_mean.csv")
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
plt.savefig(
    f"computer_science/continent_analysis/pie_test_continent_collaboration_total.png"
)
plt.close()

df = pd.read_csv("cs_eu.csv")
new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
collabs = {"EU": [], "US": [], "CN": []}
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    country_list = []
    for location in locations:
        country_list.append(location["country"])
    for i in range(len(country_list)):
        for j in range(i + 1, len(country_list)):
            if (country_list[i] in collabs.keys()) and (
                country_list[j] in collabs.keys()
            ):
                collabs[country_list[i]] += [country_list[j]]
                collabs[country_list[j]] += [country_list[i]]
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
new_df.groupby(["country", "collaboration"]).sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
    layout=(-1, 3),
)
fig.suptitle("Collaboration by Country", fontsize=16, fontweight="bold")
plt.savefig(f"computer_science/country_analysis/pie_country_collaboration_total.png")
plt.close()

countries = ["EU", "US", "CN"]
df = pd.read_csv("cs_eu.csv")
new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
collabs = {country: [] for country in countries}
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    country_list = []
    for location in locations:
        country_code = location["country"]
        if country_code in countries:
            country_list.append(country_code)
    for i in range(len(country_list)):
        for j in range(i + 1, len(country_list)):
            collabs[country_list[i]] += [country_list[j]]
            collabs[country_list[j]] += [country_list[i]]
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
new_df.groupby(["country", "collaboration"]).sum().unstack().plot(
    kind="bar", y="number"
)
plt.savefig(f"computer_science/country_analysis/bar_country_collaboration_total.png")
plt.close()

countries = ["EU", "US", "CN"]

us_collaborations = 0
eu_collaborations = 0
cn_collaborations = 0
us_eu_collaborations = 0
us_cn_collaborations = 0
eu_cn_collaborations = 0
eu_cn_us_collaborations = 0
df = pd.read_csv("cs_eu.csv")
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    country_list = []
    for location in locations:
        country_code = location["country"]
        if country_code in countries:
            country_list.append(country_code)
    if "US" in set(country_list):
        us_collaborations += 1
        if "CN" in country_list and "EU" in country_list:
            eu_cn_us_collaborations += 1
        if "CN" in country_list:
            us_cn_collaborations += 1
        if "EU" in country_list:
            us_eu_collaborations += 1
    if "CN" in country_list:
        cn_collaborations += 1
        if "EU" in country_list:
            eu_cn_collaborations += 1
    if "EU" in country_list:
        eu_collaborations += 1
with open(
    "computer_science/country_analysis/country_collaboration_cn_us_eu_percentage_total.txt",
    "w",
) as file:
    file.write(f"{collaboration_type}\n")
    file.write(
        f"CN - US collaboration represents {(us_cn_collaborations / us_collaborations) * 100:.2f}% of total US collaborations\n"
    )
    file.write(
        f"CN - US collaboration represents {(us_cn_collaborations / cn_collaborations) * 100:.2f}% of total Chinese collaborations\n"
    )
    file.write(
        f"CN - EU collaboration represents {(eu_cn_collaborations / cn_collaborations) * 100:.2f}% of total CN collaborations\n"
    )
    file.write(
        f"CN - EU collaboration represents {(eu_cn_collaborations / eu_collaborations) * 100:.2f}% of total EU collaborations\n"
    )
    file.write(
        f"EU - US collaboration represents {(us_eu_collaborations / us_collaborations) * 100:.2f}% of total US collaborations\n"
    )
    file.write(
        f"EU - US collaboration represents {(us_eu_collaborations / eu_collaborations) * 100:.2f}% of total EU collaborations\n"
    )

with open(
    "computer_science/country_analysis/country_collaboration_cn_us_eu_total.txt", "w"
) as file:
    file.write(f"{collaboration_type}\n")
    file.write(f"CN - US collaboration {(us_cn_collaborations)}\n")
    file.write(f"CN - EU collaboration {(eu_cn_collaborations)}\n")
    file.write(f"EU - US collaboration {(us_eu_collaborations)}\n")
    file.write(f"EU - US - CN collaboration {(eu_cn_us_collaborations)}\n")


# Define the data
us_data = [us_collaborations, us_eu_collaborations, us_cn_collaborations]
eu_data = [us_eu_collaborations, eu_collaborations, eu_cn_collaborations]
cn_data = [eu_cn_collaborations, us_cn_collaborations, cn_collaborations]

# Define the x-axis labels
labels = ["US Collaborations", "EU Collaborations", "CN Collaborations"]

# Define the x-axis locations for each group of bars
x_us = [0, 4, 8]
x_eu = [1, 5, 9]
x_cn = [2, 6, 10]

# Plot the bars
plt.bar(x_us, us_data, color="blue", width=0.8, label="US")
plt.bar(x_eu, eu_data, color="red", width=0.8, label="EU")
plt.bar(x_cn, cn_data, color="green", width=0.8, label="CN")

# Add the x-axis labels and tick marks
plt.xticks([1.5, 5.5, 9.5], labels)
plt.xlabel("Collaboration Type")
plt.ylabel("Number of Collaborations")

# Add a legend
plt.legend()

# Show the plot
plt.savefig(
    f"computer_science/country_analysis/bar_country_collaboration_cn_us_eu_total.png"
)
plt.close()

# Define the data
us_data = [
    (us_collaborations / us_collaborations) * 100,
    (us_eu_collaborations / us_collaborations) * 100,
    (us_cn_collaborations / us_collaborations) * 100,
]
eu_data = [
    (us_eu_collaborations / eu_collaborations) * 100,
    (eu_collaborations / eu_collaborations) * 100,
    (us_cn_collaborations / eu_collaborations) * 100,
]
cn_data = [
    (us_cn_collaborations / cn_collaborations) * 100,
    (eu_cn_collaborations / cn_collaborations) * 100,
    (cn_collaborations / cn_collaborations) * 100,
]

# Define the x-axis labels
labels = ["US Collaborations", "EU Collaborations", "CN Collaborations"]

# Define the x-axis locations for each group of bars
x_us = [0, 4, 8]
x_eu = [1, 5, 9]
x_cn = [2, 6, 10]

# Plot the bars
plt.bar(x_us, us_data, color="blue", width=0.8, label="US")
plt.bar(x_eu, eu_data, color="red", width=0.8, label="EU")
plt.bar(x_cn, cn_data, color="green", width=0.8, label="CN")

# Add the x-axis labels and tick marks
plt.xticks([1.5, 5.5, 9.5], labels)
plt.xlabel("Collaboration Type")
plt.ylabel("Number of Collaborations")

# Add a legend
plt.legend()

# Show the plot
plt.savefig(
    f"computer_science/country_analysis/bar_country_collaboration_cn_us_eu_percent_total.png"
)
plt.close()

# endregion


## DEVELOPMENT ANALYSIS

# region development

Path("computer_science/development_analysis/").mkdir(parents=True, exist_ok=True)
df_international = df[df["international"] == True]
unique_dev_types = df["no_dev"].unique()
# df = df[df["no_dev"] == True]

df_international["dist_trunc"] = round(df["distance"], 0)
with open(
    "computer_science/development_analysis/correlation_max_distance_compared_to_hdi_by_type.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Spearman test for "
            + collaboration_type
            + " - "
            + str(
                stats.spearmanr(
                    df_international[df_international["type"] == collaboration_type][
                        "mean_index"
                    ],
                    df_international[df_international["type"] == collaboration_type][
                        "distance"
                    ],
                )
            )
        )
        f.write("\n")
        f.write(
            "Pearson test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    df_international[df_international["type"] == collaboration_type][
                        "mean_index"
                    ],
                    df_international[df_international["type"] == collaboration_type][
                        "distance"
                    ],
                )
            )
        )
        f.write("\n")
with open(
    "computer_science/development_analysis/correlation_mean_distance_compared_to_hdi_by_type.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Spearman test for "
            + collaboration_type
            + " - "
            + str(
                stats.spearmanr(
                    df_international[df_international["type"] == collaboration_type][
                        "mean_index"
                    ],
                    df_international[df_international["type"] == collaboration_type][
                        "mean_distance"
                    ],
                )
            )
        )
        f.write("\n")
        f.write(
            "Pearson test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    df_international[df_international["type"] == collaboration_type][
                        "mean_index"
                    ],
                    df_international[df_international["type"] == collaboration_type][
                        "mean_distance"
                    ],
                )
            )
        )
        f.write("\n")
with open(
    "computer_science/development_analysis/correlation_citations_compared_to_hdi_by_type.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Spearman test for "
            + collaboration_type
            + " - "
            + str(
                stats.spearmanr(
                    df_international[df_international["type"] == collaboration_type][
                        "mean_index"
                    ],
                    df_international[df_international["type"] == collaboration_type][
                        "citations"
                    ],
                )
            )
        )
        f.write("\n")
        f.write(
            "Pearson test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    df_international[df_international["type"] == collaboration_type][
                        "mean_index"
                    ],
                    df_international[df_international["type"] == collaboration_type][
                        "citations"
                    ],
                )
            )
        )
        f.write("\n")

# Descriptive statistics
df_international.groupby("no_dev")["distance"].describe().to_csv(
    "computer_science/development_analysis/describe_max_distance_by_type_by_developement.csv"
)
df_international.groupby("no_dev")["mean_distance"].describe().to_csv(
    "computer_science/development_analysis/describe_mean_distance_by_type_by_developement.csv"
)
df_international.groupby("no_dev")["citations"].describe().to_csv(
    "computer_science/development_analysis/describe_citations_by_type_by_developement.csv"
)

# Normality check
# It does not come from normal distribution

# Kruskal test - Statistical diference between groups
with open(
    "computer_science/development_analysis/kruskal_max_distance_by_development.txt", "w"
) as f:
    f.write(
        "Kruskal Test for Max distance by type"
        + str(
            stats.kruskal(
                df_international[df_international["no_dev"] == True]["distance"],
                df_international[df_international["no_dev"] == False]["distance"],
            )
        )
    )
with open(
    "computer_science/development_analysis/kruskal_mean_distance_by_development.txt",
    "w",
) as f:
    f.write(
        "Kruskal Test for Mean distance by type"
        + str(
            stats.kruskal(
                df[df["no_dev"] == True]["mean_distance"],
                df[df["no_dev"] == False]["mean_distance"],
            )
        )
    )
with open(
    "computer_science/development_analysis/kruskal_citations_by_development.txt", "w"
) as f:
    f.write(
        "Kruskal Test for Mean distance by type"
        + str(
            stats.kruskal(
                df_international[df_international["no_dev"] == True]["citations"],
                df_international[df_international["no_dev"] == False]["citations"],
            )
        )
    )

# Pearson test and Spearman test- correlation coeficient
df_international_max_trunc = (
    df_international.groupby(["no_dev", "dist_trunc"]).size().reset_index(name="count")
)
with open(
    "computer_science/development_analysis/correlation_max_distance_compared_to_count_by_development.txt",
    "w",
) as f:
    for dev_type in unique_dev_types:
        if dev_type == True:
            dev_type_name = "NO developed countries"
        else:
            dev_type_name = "Developed countries"
        f.write(
            "Spearman test for "
            + dev_type_name
            + " - "
            + str(
                stats.spearmanr(
                    df_international_max_trunc[
                        df_international_max_trunc["no_dev"] == dev_type
                    ]["dist_trunc"],
                    df_international_max_trunc[
                        df_international_max_trunc["no_dev"] == dev_type
                    ]["count"],
                )
            )
        )
        f.write("\n")
        f.write(
            "Pearson test for "
            + dev_type_name
            + " - "
            + str(
                stats.pearsonr(
                    df_international_max_trunc[
                        df_international_max_trunc["no_dev"] == dev_type
                    ]["dist_trunc"],
                    df_international_max_trunc[
                        df_international_max_trunc["no_dev"] == dev_type
                    ]["count"],
                )
            )
        )
        f.write("\n")

df_international_mean_trunc = (
    df_international.groupby(["no_dev", "mean_dist_trunc"])
    .size()
    .reset_index(name="count")
)
with open(
    "computer_science/development_analysis/correlation_mean_distance_compared_to_count_by_development.txt",
    "w",
) as f:
    for dev_type in unique_dev_types:
        if dev_type == True:
            dev_type_name = "NO developed countries"
        else:
            dev_type_name = "Developed countries"
        f.write(
            "Spearman test for "
            + dev_type_name
            + " - "
            + str(
                stats.spearmanr(
                    df_international_mean_trunc[
                        df_international_mean_trunc["no_dev"] == dev_type
                    ]["mean_dist_trunc"],
                    df_international_mean_trunc[
                        df_international_mean_trunc["no_dev"] == dev_type
                    ]["count"],
                )
            )
        )
        f.write("\n")
        f.write(
            "Pearson test for "
            + dev_type_name
            + " - "
            + str(
                stats.pearsonr(
                    df_international_mean_trunc[
                        df_international_mean_trunc["no_dev"] == dev_type
                    ]["mean_dist_trunc"],
                    df_international_mean_trunc[
                        df_international_mean_trunc["no_dev"] == dev_type
                    ]["count"],
                )
            )
        )
        f.write("\n")

with open(
    "computer_science/development_analysis/correlation_citations_compared_to_hdi_by_development.txt",
    "w",
) as f:
    for dev_type in unique_dev_types:
        if dev_type == True:
            dev_type_name = "NO developed countries"
        else:
            dev_type_name = "Developed countries"
        f.write(
            "Spearman test for "
            + dev_type_name
            + " - "
            + str(
                stats.spearmanr(
                    df_international[df_international["no_dev"] == dev_type][
                        "citations"
                    ],
                    df_international[df_international["no_dev"] == dev_type][
                        "mean_index"
                    ],
                )
            )
        )
        f.write("\n")
        f.write(
            "Pearson test for "
            + dev_type_name
            + " - "
            + str(
                stats.pearsonr(
                    df_international[df_international["no_dev"] == dev_type][
                        "citations"
                    ],
                    df_international[df_international["no_dev"] == dev_type][
                        "mean_index"
                    ],
                )
            )
        )
        f.write("\n")

# Index by citations
sns.lmplot(
    x="mean_index",
    y="citations",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.xlabel("Mean Human Development Index (HDI)")
plt.ylabel("Number of Citations")
plt.title("HDI compared to Citations by Type")
plt.savefig(
    f"computer_science/development_analysis/scatter_hdi_compared_to_citations_by_type.png"
)
plt.close()

# Index by distance
sns.lmplot(
    x="mean_index",
    y="distance",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.xlabel("Mean Human Development Index (HDI)")
plt.ylabel("Maximum Distance")
plt.title("HDI compared to Maximum Distance by Type")
plt.savefig(
    f"computer_science/development_analysis/scatter_hdi_compared_to_max_distance_by_type.png"
)
plt.close()

sns.lmplot(
    x="mean_index",
    y="mean_distance",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.xlabel("Mean Human Development Index (HDI)")
plt.ylabel("Mean Distance")
plt.title("HDI compared to Mean Distance by Type")
plt.savefig(
    f"computer_science/development_analysis/scatter_hdi_compared_to_mean_distance_by_type.png"
)
plt.close()

# citations by distance
sns.lmplot(
    x="citations",
    y="distance",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.xlabel("Number of Citations")
plt.ylabel("Maximum Distance")
plt.title("Citations compared to Maximum Distance by Type")
plt.savefig(
    f"computer_science/development_analysis/scatter_citations_compared_to_max_distance_by_type.png"
)
plt.close()

sns.lmplot(
    x="citations",
    y="mean_distance",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.xlabel("Number of Citations")
plt.ylabel("Mean Distance")
plt.title("Citations compared to Mean Distance by Type")
plt.savefig(
    f"computer_science/development_analysis/scatter_citations_compared_to_mean_distance_by_type.png"
)
plt.close()


# mean distance by development
ax = (
    df_international.groupby(["no_dev", "type"])["distance"]
    .mean()
    .unstack()
    .plot(kind="bar", figsize=(10, 8))
)
ax.set_xlabel("Development")
ax.set_ylabel("Mean distance")
ax.set_title("Mean distance by development and type")
plt.savefig(
    f"computer_science/development_analysis/bar_max_distance_by_developement_by_type.png"
)
plt.close()

ax = (
    df_international.groupby(["no_dev", "type"])["mean_distance"]
    .mean()
    .unstack()
    .plot(kind="bar", figsize=(10, 8))
)
ax.set_xlabel("Development")
ax.set_ylabel("Mean distance")
ax.set_title("Mean distance by development and type")
plt.savefig(
    f"computer_science/development_analysis/bar_mean_distance_by_developement_by_type.png"
)
plt.close()

# mean citations by development
ax = (
    df_international.groupby(["no_dev", "type"])["citations"]
    .mean()
    .unstack()
    .plot(kind="bar", figsize=(10, 8))
)
ax.set_xlabel("Development")
ax.set_ylabel("Mean citations")
ax.set_title("Mean citations by development and type")
plt.savefig(
    f"computer_science/development_analysis/bar_citations_by_developement_by_type.png"
)
plt.close()
# Create and save a CSV file with descriptive statistics for citations and distance by development and type
df_international[["type", "citations", "no_dev", "distance", "mean_distance"]].groupby(
    ["type", "no_dev"]
).describe().describe().to_csv(
    "computer_science/development_analysis/describe_citations_and_max_distance_and_mean_distance_by_development_by_type.csv"
)

# Pie chart showing the count of citations by development and type
no_dev_df = df_international[["type", "citations", "no_dev", "distance"]]
no_dev_df.groupby(["type", "no_dev"])["citations"].count().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.title("Count of Citations by Development and Type")
plt.ylabel("")
plt.savefig(
    f"computer_science/development_analysis/pie_citations_count_by_developement_by_type.png"
)
plt.close()

# Pie chart showing the sum of citations by development and type
no_dev_df.groupby(["type", "no_dev"])["citations"].sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.title("Sum of Citations by Development and Type")
plt.ylabel("")
plt.savefig(
    f"computer_science/development_analysis/pie_citations_sum_by_developement_by_type.png"
)
plt.close()

with open(
    "computer_science/development_analysis/kruskal_citations_by_developemnt_and_type.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Kruskal Test for citations by development for "
            + collaboration_type
            + " "
            + str(
                stats.kruskal(
                    df_international[
                        (df_international["type"] == collaboration_type)
                        & (df_international["no_dev"] == False)
                    ]["citations"],
                    df_international[
                        (df_international["type"] == collaboration_type)
                        & (df_international["no_dev"] == True)
                    ]["citations"],
                )
            )
        )

no_dev_df = df[["citations", "no_dev", "distance", "international"]]
df[["no_dev", "citations", "international"]].groupby(
    ["no_dev", "international"]
).describe().to_csv(
    "computer_science/development_analysis/describe_citations_by_international_by_development.csv"
)
# Plotting pie charts for citation count and sum by development and international collaboration
no_dev_df.groupby(["no_dev", "international"])["citations"].count().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.title(
    "Percentage of Citations Count by Development and International Collaboration"
)
plt.xlabel("Development and International Collaboration")
plt.ylabel("Percentage of Citations Count")
plt.savefig(
    f"computer_science/development_analysis/pie_citations_count_by_developement_by_international.png"
)
plt.close()

no_dev_df.groupby(["no_dev", "international"])["citations"].sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.title("Percentage of Citations Sum by Development and International Collaboration")
plt.xlabel("Development and International Collaboration")
plt.ylabel("Percentage of Citations Sum")
plt.savefig(
    f"computer_science/development_analysis/pie_citations_sum_by_developement_by_international.png"
)
plt.close()

# endregion

## SCOPE ANALYSIS

Path("computer_science/scope_analysis/").mkdir(parents=True, exist_ok=True)
# region scope

# Descriptive statistics


def f(row):
    if row["ratio"] == 0.5:
        val = "half"
    elif row["ratio"] > 0.5:
        val = "com"
    else:
        val = "edu"
    return val


ratio_df = df[df["ratio"] < 1]
ratio_df = ratio_df[ratio_df["ratio"] > 0]

ratio_df["ratio_type"] = ratio_df.apply(f, axis=1)

ratio_df[["ratio_type", "citations"]].groupby("ratio_type").describe().to_csv(
    "computer_science/scope_analysis/describe_citations_by_ratio_type.csv"
)
ratio_df[["ratio_type", "distance"]].groupby("ratio_type").describe().to_csv(
    "computer_science/scope_analysis/describe_max_distance_by_ratio_type.csv"
)
ratio_df[["ratio_type", "mean_distance"]].groupby("ratio_type").describe().to_csv(
    "computer_science/scope_analysis/describe_mean_distance_by_ratio_type.csv"
)

with open(
    "computer_science/scope_analysis/kruskal_max_distance_by_ratio_type.txt", "w"
) as f:
    f.write(
        "Kruskal Test for Max distance by ratio type"
        + str(
            stats.kruskal(
                ratio_df[ratio_df["ratio_type"] == "edu"]["distance"],
                ratio_df[ratio_df["ratio_type"] == "com"]["distance"],
                ratio_df[ratio_df["ratio_type"] == "half"]["distance"],
            )
        )
    )
with open(
    "computer_science/scope_analysis/kruskal_mean_distance_by_ratio_type.txt", "w"
) as f:
    f.write(
        "Kruskal Test for Mean distance by ratio type"
        + str(
            stats.kruskal(
                ratio_df[ratio_df["ratio_type"] == "edu"]["mean_distance"],
                ratio_df[ratio_df["ratio_type"] == "com"]["mean_distance"],
                ratio_df[ratio_df["ratio_type"] == "half"]["mean_distance"],
            )
        )
    )
with open(
    "computer_science/scope_analysis/kruskal_citations_by_ratio_type.txt", "w"
) as f:
    f.write(
        "Kruskal Test for citations by ratio type"
        + str(
            stats.kruskal(
                ratio_df[ratio_df["ratio_type"] == "edu"]["citations"],
                ratio_df[ratio_df["ratio_type"] == "com"]["citations"],
                ratio_df[ratio_df["ratio_type"] == "half"]["citations"],
            )
        )
    )

# Scatter plot with lmplot
sns.lmplot(x="ratio", y="citations", data=ratio_df, scatter=False)
plt.xlabel("Ratio")
plt.ylabel("Citations")
plt.title("Scatter plot of Citations vs Ratio")
plt.savefig("computer_science/scope_analysis/scatter_citations_ratio.png")
plt.close()

# Bar plot of max distance by ratio type
ratio_df.groupby("ratio_type")["distance"].mean().plot(kind="bar")
plt.xlabel("Ratio Type")
plt.ylabel("Max Distance")
plt.title("Mean Max Distance by Ratio Type")
plt.savefig("computer_science/scope_analysis/bar_max_distance_by_ratio.png")
plt.close()

# Bar plot of mean distance by ratio type
ratio_df.groupby("ratio_type")["mean_distance"].mean().plot(kind="bar")
plt.xlabel("Ratio Type")
plt.ylabel("Mean Distance")
plt.title("Mean Distance by Ratio Type")
plt.savefig("computer_science/scope_analysis/bar_mean_distance_by_ratio.png")
plt.close()

# Bar plot of citations by ratio type
ratio_df.groupby("ratio_type")["citations"].mean().plot(kind="bar")
plt.xlabel("Ratio Type")
plt.ylabel("Citations")
plt.title("Mean Citations by Ratio Type")
plt.savefig("computer_science/scope_analysis/bar_citations_by_ratio.png")
plt.close()

# Pie chart of count of citations by ratio type
ratio_df.groupby("ratio_type")["citations"].count().plot.pie(
    subplots=True, autopct="%1.1f%%", legend=False, startangle=90, figsize=(10, 7)
)
plt.ylabel("")
plt.title("Count of Citations by Ratio Type")
plt.savefig("computer_science/scope_analysis/pie_count_citations_by_ratio.png")
plt.close()

# Pie chart of sum of citations by ratio type
ratio_df.groupby("ratio_type")["citations"].sum().plot.pie(
    subplots=True, autopct="%1.1f%%", legend=False, startangle=90, figsize=(10, 7)
)
plt.ylabel("")
plt.title("Sum of Citations by Ratio Type")
plt.savefig("computer_science/scope_analysis/pie_sum_citations_by_ratio.png")
plt.close()
# endregion scope

## TOPIC ANALYSIS

Path("computer_science/topic_analysis/").mkdir(parents=True, exist_ok=True)
"""hm_df = pd.DataFrame(
    {
        "work": str,
        "continent": [],
        "concept": [],
        "year":int,
        "no_dev":bool,
    }
)
continent_concept_list = []
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    continents = []
    for continent in locations:
        continents.append(continent["continent"])
    for continent in set(continents):
        continent = "NAA" if continent == "NA" else continent
        concepts = literal_eval(row.concepts)
        for concept in concepts:
            continent_concept_list.append([row.work, continent, concept, row.year, row.no_dev, row.type])
hm_df = pd.DataFrame(continent_concept_list, columns = ['work','continent', 'concept', 'year', 'no_dev', 'type'])
hm_df.to_csv("test_concepts.csv")"""
hm_df_full = pd.read_csv("test_concepts.csv")
unique_dev_types = df["no_dev"].unique()

for collaboration_type in unique_collaboration_types:
    for dev_type in unique_dev_types:
        if dev_type == True:
            dev_type_name = "no_developed"
        else:
            dev_type_name = "developed"
        test = (
            hm_df_full.groupby("concept")["work"]
            .count()
            .reset_index(name="count")
            .sort_values(by=["count"], ascending=False)
            .head(11)
        )
        test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
        new_df = hm_df_full.loc[hm_df_full["concept"].isin(test.concept.to_list())]
        means = (
            new_df.groupby(["no_dev", "concept", "year", "type"])["work"]
            .count()
            .reset_index(name="count")
        )
        means = means[
            (means["no_dev"] == dev_type) & (means["type"] == collaboration_type)
        ]
        sns.lineplot(data=means, x="year", y="count", hue="concept")
        plt.savefig(
            f"computer_science/topic_analysis/line_topics_by_year_by_development_{dev_type_name}.png"
        )
        plt.close()


# for collaboration_type in unique_collaboration_types:
# for developement_type in unique_dev_types:
unique_continents = ["NAA", "OC", "EU", "AS", "AF", "SA"]
for unique_continent in unique_continents:
    hm_df_full = pd.read_csv("test_concepts.csv")
    test = (
        hm_df_full.groupby("concept")["work"]
        .count()
        .reset_index(name="count")
        .sort_values(by=["count"], ascending=False)
        .head(11)
    )
    test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
    new_df = hm_df_full.loc[hm_df_full["concept"].isin(test.concept.to_list())]
    means_full = (
        new_df.groupby(["continent", "concept", "year"])["work"]
        .count()
        .reset_index(name="count")
    )
    means = means_full[
        (means_full["continent"] == unique_continent)
    ]  # & (means["type"]=="mixed")
    sns.lineplot(data=means, x="year", y="count", hue="concept")
    plt.savefig(
        f"computer_science/topic_analysis/line_topics_by_year_by_contient_{unique_continent}.png"
    )
    plt.close()

"""
countries = ["EU", "US", "CN"]
df = pd.read_csv("cs_eu.csv")
new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
collabs = {country: [] for country in countries}
continent_concept_list = []
hm_df = pd.DataFrame(
    {
        "work": str,
        "country": [],
        "concept": [],
        "year":int,
        "no_dev":bool,
    }
)
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    country_list= []
    for location in locations:
        country_code = location["country"]
        if country_code in countries:
            country_list.append(country_code)
    concepts = literal_eval(row.concepts)
    for concept in concepts:
        for country in country_list:
            continent_concept_list.append([row.work, country, concept, row.year, row.no_dev, row.type])
hm_df = pd.DataFrame(continent_concept_list, columns = ['work','country', 'concept', 'year', 'no_dev', 'type'])
hm_df.to_csv("test_concepts_eu_us_cn.csv")"""


unique_continents = ["CN", "US", "EU"]
for unique_continent in unique_continents:
    hm_df_full = pd.read_csv("test_concepts_eu_us_cn.csv")
    test = (
        hm_df_full.groupby("concept")["work"]
        .count()
        .reset_index(name="count")
        .sort_values(by=["count"], ascending=False)
        .head(11)
    )
    test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
    new_df = hm_df_full.loc[hm_df_full["concept"].isin(test.concept.to_list())]
    means_full = (
        new_df.groupby(["continent", "concept", "year"])["work"]
        .count()
        .reset_index(name="count")
    )
    means = means_full[
        (means_full["continent"] == unique_continent)
    ]  # & (means["type"]=="mixed")
    sns.lineplot(data=means, x="year", y="count", hue="concept")
    plt.savefig(
        f"computer_science/topic_analysis/line_topics_by_year_by_country_{unique_continent}.png"
    )
    plt.close()

for i, continent1 in enumerate(unique_continents):
    for continent2 in unique_continents[i + 1 :]:
        hm_df_full = pd.read_csv("test_concepts_eu_us_cn.csv")
        test = (
            hm_df_full.groupby("concept")["work"]
            .count()
            .reset_index(name="count")
            .sort_values(by=["count"], ascending=False)
            .head(11)
        )
        test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
        new_df = hm_df_full.loc[hm_df_full["concept"].isin(test.concept.to_list())]
        means_full = (
            new_df.groupby(["year", "concept"])
            .apply(
                lambda x: x[x["continent"].isin([continent1, continent2])][
                    "work"
                ].count()
            )
            .reset_index(name="count")
        )
        means_full.rename(columns={"level_2": "collaboration"}, inplace=True)
        means_full["collaboration"] = f"{continent1}-{continent2}"
        sns.lineplot(data=means_full, x="year", y="count", hue="concept")
        plt.title(
            f"Topics per year by concept for {continent1}-{continent2} collaborations"
        )
        plt.savefig(
            f"computer_science/topic_analysis/line_topics_by_year_{continent1}_{continent2}.png"
        )
        plt.close()
