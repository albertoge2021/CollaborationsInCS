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

## DEVELOPMENT ANALYSIS

#region development

Path("computer_science/development_analysis/").mkdir(parents=True, exist_ok=True)
df_international = df[df["international"] == True]
unique_dev_types = df["no_dev"].unique()
#df = df[df["no_dev"] == True]

df_international["dist_trunc"] = round(df["distance"], 0)
with open('computer_science/development_analysis/correlation_max_distance_compared_to_hdi_by_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.spearmanr(df_international[df_international['type'] == collaboration_type]['mean_index'], df_international[df_international['type'] == collaboration_type]['distance'])))
        f.write('\n')
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df_international[df_international['type'] == collaboration_type]['mean_index'], df_international[df_international['type'] == collaboration_type]['distance'])))
        f.write('\n')
with open('computer_science/development_analysis/correlation_mean_distance_compared_to_hdi_by_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.spearmanr(df_international[df_international['type'] == collaboration_type]['mean_index'], df_international[df_international['type'] == collaboration_type]['mean_distance'])))
        f.write('\n')
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df_international[df_international['type'] == collaboration_type]['mean_index'], df_international[df_international['type'] == collaboration_type]['mean_distance'])))
        f.write('\n')
with open('computer_science/development_analysis/correlation_citations_compared_to_hdi_by_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.spearmanr(df_international[df_international['type'] == collaboration_type]['mean_index'], df_international[df_international['type'] == collaboration_type]['citations'])))
        f.write('\n')
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df_international[df_international['type'] == collaboration_type]['mean_index'], df_international[df_international['type'] == collaboration_type]['citations'])))
        f.write('\n')

# Descriptive statistics
df_international.groupby('no_dev')['distance'].describe().to_csv("computer_science/development_analysis/describe_max_distance_by_type_by_developement.csv")
df_international.groupby('no_dev')['mean_distance'].describe().to_csv("computer_science/development_analysis/describe_mean_distance_by_type_by_developement.csv")
df_international.groupby('no_dev')['citations'].describe().to_csv("computer_science/development_analysis/describe_citations_by_type_by_developement.csv")

# Normality check
# It does not come from normal distribution

# Kruskal test - Statistical diference between groups
with open('computer_science/development_analysis/kruskal_max_distance_by_development.txt', 'w') as f:
    f.write("Kruskal Test for Max distance by type" + str(stats.kruskal(df_international[df_international['no_dev'] == True]['distance'], df_international[df_international['no_dev'] == False]['distance'])))
with open('computer_science/development_analysis/kruskal_mean_distance_by_development.txt', 'w') as f:
    f.write("Kruskal Test for Mean distance by type" + str(stats.kruskal(df[df['no_dev'] == True]['mean_distance'], df[df['no_dev'] == False]['mean_distance'])))
with open('computer_science/development_analysis/kruskal_citations_by_development.txt', 'w') as f:
    f.write("Kruskal Test for Mean distance by type" + str(stats.kruskal(df_international[df_international['no_dev'] == True]['citations'], df_international[df_international['no_dev'] == False]['citations'])))

# Pearson test and Spearman test- correlation coeficient
df_international_max_trunc = df_international.groupby(["no_dev", "dist_trunc"]).size().reset_index(name="count")
with open('computer_science/development_analysis/correlation_max_distance_compared_to_count_by_development.txt', 'w') as f:
    for dev_type in unique_dev_types:
        if dev_type == True:
            dev_type_name = "NO developed selected_countries"
        else:
            dev_type_name = "Developed selected_countries"
        f.write("Spearman test for "+dev_type_name +" - " + str(stats.spearmanr(df_international_max_trunc[df_international_max_trunc['no_dev'] == dev_type]['dist_trunc'], df_international_max_trunc[df_international_max_trunc['no_dev'] == dev_type]['count'])))
        f.write('\n')
        f.write("Pearson test for "+dev_type_name +" - " + str(stats.pearsonr(df_international_max_trunc[df_international_max_trunc['no_dev'] == dev_type]['dist_trunc'], df_international_max_trunc[df_international_max_trunc['no_dev'] == dev_type]['count'])))
        f.write('\n')

df_international_mean_trunc = df_international.groupby(["no_dev", "mean_dist_trunc"]).size().reset_index(name="count")
with open('computer_science/development_analysis/correlation_mean_distance_compared_to_count_by_development.txt', 'w') as f:
    for dev_type in unique_dev_types:
        if dev_type == True:
            dev_type_name = "NO developed selected_countries"
        else:
            dev_type_name = "Developed selected_countries"
        f.write("Spearman test for "+dev_type_name +" - " + str(stats.spearmanr(df_international_mean_trunc[df_international_mean_trunc['no_dev'] == dev_type]['mean_dist_trunc'], df_international_mean_trunc[df_international_mean_trunc['no_dev'] == dev_type]['count'])))
        f.write('\n')
        f.write("Pearson test for "+dev_type_name +" - " + str(stats.pearsonr(df_international_mean_trunc[df_international_mean_trunc['no_dev'] == dev_type]['mean_dist_trunc'], df_international_mean_trunc[df_international_mean_trunc['no_dev'] == dev_type]['count'])))
        f.write('\n')

with open('computer_science/development_analysis/correlation_citations_compared_to_hdi_by_development.txt', 'w') as f:
    for dev_type in unique_dev_types:
        if dev_type == True:
            dev_type_name = "NO developed selected_countries"
        else:
            dev_type_name = "Developed selected_countries"
        f.write("Spearman test for "+dev_type_name +" - " + str(stats.spearmanr(df_international[df_international['no_dev'] == dev_type]['citations'], df_international[df_international['no_dev'] == dev_type]['mean_index'])))
        f.write('\n')
        f.write("Pearson test for "+dev_type_name +" - " + str(stats.pearsonr(df_international[df_international['no_dev'] == dev_type]['citations'], df_international[df_international['no_dev'] == dev_type]['mean_index'])))
        f.write('\n')

#Index by citations
sns.lmplot(
    x="mean_index",
    y="citations",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.xlabel('Mean Human Development Index (HDI)') 
plt.ylabel('Number of Citations') 
plt.title('HDI compared to Citations by Type')
plt.savefig(f'computer_science/development_analysis/scatter_hdi_compared_to_citations_by_type.png')
plt.close()

#Index by distance
sns.lmplot(
    x="mean_index",
    y="distance",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.xlabel('Mean Human Development Index (HDI)') 
plt.ylabel('Maximum Distance') 
plt.title('HDI compared to Maximum Distance by Type')
plt.savefig(f'computer_science/development_analysis/scatter_hdi_compared_to_max_distance_by_type.png')
plt.close()

sns.lmplot(
    x="mean_index",
    y="mean_distance",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.xlabel('Mean Human Development Index (HDI)') 
plt.ylabel('Mean Distance') 
plt.title('HDI compared to Mean Distance by Type')
plt.savefig(f'computer_science/development_analysis/scatter_hdi_compared_to_mean_distance_by_type.png')
plt.close()

#citations by distance
sns.lmplot(
    x="citations",
    y="distance",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.xlabel('Number of Citations') 
plt.ylabel('Maximum Distance') 
plt.title('Citations compared to Maximum Distance by Type')
plt.savefig(f'computer_science/development_analysis/scatter_citations_compared_to_max_distance_by_type.png')
plt.close()

sns.lmplot(
    x="citations",
    y="mean_distance",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.xlabel('Number of Citations') 
plt.ylabel('Mean Distance') 
plt.title('Citations compared to Mean Distance by Type')
plt.savefig(f'computer_science/development_analysis/scatter_citations_compared_to_mean_distance_by_type.png')
plt.close()


# mean distance by development
ax = df_international.groupby(['no_dev', 'type'])['distance'].mean().unstack().plot(kind='bar', figsize=(10,8))
ax.set_xlabel('Development')
ax.set_ylabel('Mean distance')
ax.set_title('Mean distance by development and type')
plt.savefig(f'computer_science/development_analysis/bar_max_distance_by_developement_by_type.png')
plt.close()

ax = df_international.groupby(['no_dev', 'type'])['mean_distance'].mean().unstack().plot(kind='bar', figsize=(10,8))
ax.set_xlabel('Development')
ax.set_ylabel('Mean distance')
ax.set_title('Mean distance by development and type')
plt.savefig(f'computer_science/development_analysis/bar_mean_distance_by_developement_by_type.png')
plt.close()

# mean citations by development
ax = df_international.groupby(['no_dev', 'type'])['citations'].mean().unstack().plot(kind='bar', figsize=(10,8))
ax.set_xlabel('Development')
ax.set_ylabel('Mean citations')
ax.set_title('Mean citations by development and type')
plt.savefig(f'computer_science/development_analysis/bar_citations_by_developement_by_type.png')
plt.close()
# Create and save a CSV file with descriptive statistics for citations and distance by development and type
df_international[["type","citations", "no_dev", "distance","mean_distance"]].groupby(["type", "no_dev"]).describe().describe().to_csv("computer_science/development_analysis/describe_citations_and_max_distance_and_mean_distance_by_development_by_type.csv")

# Pie chart showing the count of citations by development and type
no_dev_df = df_international[["type", "citations", "no_dev", "distance"]]
no_dev_df.groupby(["type", "no_dev"])['citations'].count().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.title("Count of Citations by Development and Type")
plt.ylabel("")
plt.savefig(f'computer_science/development_analysis/pie_citations_count_by_developement_by_type.png')
plt.close()

# Pie chart showing the sum of citations by development and type
no_dev_df.groupby(["type", "no_dev"])['citations'].sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.title("Sum of Citations by Development and Type")
plt.ylabel("")
plt.savefig(f'computer_science/development_analysis/pie_citations_sum_by_developement_by_type.png')
plt.close()

with open('computer_science/development_analysis/kruskal_citations_by_developemnt_and_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Kruskal Test for citations by development for " + collaboration_type  + " " + str(stats.kruskal(df_international[(df_international['type'] == collaboration_type) & (df_international['no_dev'] == False)]['citations'], df_international[(df_international['type'] == collaboration_type) & (df_international['no_dev'] == True)]['citations'])))

no_dev_df = df[["citations", "no_dev", "distance", "international"]]
df[["no_dev","citations", "international"]].groupby(["no_dev", "international"]).describe().to_csv("computer_science/development_analysis/describe_citations_by_international_by_development.csv")
# Plotting pie charts for citation count and sum by development and international collaboration
no_dev_df.groupby(["no_dev", "international"])['citations'].count().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.title('Percentage of Citations Count by Development and International Collaboration')
plt.xlabel('Development and International Collaboration')
plt.ylabel('Percentage of Citations Count')
plt.savefig(f'computer_science/development_analysis/pie_citations_count_by_developement_by_international.png')
plt.close()

no_dev_df.groupby(["no_dev", "international"])['citations'].sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.title('Percentage of Citations Sum by Development and International Collaboration')
plt.xlabel('Development and International Collaboration')
plt.ylabel('Percentage of Citations Sum')
plt.savefig(f'computer_science/development_analysis/pie_citations_sum_by_developement_by_international.png')
plt.close()

#endregion
