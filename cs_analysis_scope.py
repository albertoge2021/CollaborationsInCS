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
