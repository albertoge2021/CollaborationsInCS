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
