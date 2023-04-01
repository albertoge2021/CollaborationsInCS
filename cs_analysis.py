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


"""# Descriptive statistics
Path("computer_science/general_analysis/").mkdir(parents=True, exist_ok=True)
df[["type","distance"]].groupby("type").describe().to_csv("computer_science/general_analysis/describe_max_distance_by_type.csv")
df[["type","mean_distance"]].groupby("type").describe().to_csv("computer_science/general_analysis/describe_mean_distance_by_type.csv")

# Normality check
# It does not come from normal distribution

# Kruskal test - Statistical diference between groups
with open('computer_science/general_analysis/kruskal_max_distance_by_type.txt', 'w') as f:
    f.write("Kruskal Test for Max distance by type" + str(stats.kruskal(df[df['type'] == "company"]['distance'], df[df['type'] == "education"]['distance'], df[df['type'] == "mixed"]['distance'])))
with open('computer_science/general_analysis/kruskal_mean_distance_by_type.txt', 'w') as f:
    f.write("Kruskal Test for Mean distance by type" + str(stats.kruskal(df[df['type'] == "company"]['mean_distance'], df[df['type'] == "education"]['mean_distance'], df[df['type'] == "mixed"]['mean_distance'])))
with open('computer_science/general_analysis/kruskal_citations_by_type.txt', 'w') as f:
    f.write("Kruskal Test for Mean distance by type" + str(stats.kruskal(df[df['type'] == "company"]['citations'], df[df['type'] == "education"]['citations'], df[df['type'] == "mixed"]['citations'])))

# Pearson test and Spearman test- correlation coeficient
unique_collaboration_types = df["type"].unique()
df["dist_trunc"] = round(df["distance"], 0)
max_df = df.groupby(["type", "dist_trunc"]).size().reset_index(name="count")
with open('computer_science/general_analysis/correlation_max_distance_compared_to_count_by_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(max_df[max_df['type'] == collaboration_type]['dist_trunc'], max_df[max_df['type'] == collaboration_type]['count'])))
        f.write('\n')
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.pearsonr(max_df[max_df['type'] == collaboration_type]['dist_trunc'], max_df[max_df['type'] == collaboration_type]['count'])))
        f.write('\n')

df["mean_dist_trunc"] = round(df["mean_distance"], 0)
mean_df = df.groupby(["type", "mean_dist_trunc"]).size().reset_index(name="count")
with open('computer_science/general_analysis/correlation_mean_distance_compared_to_count_by_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(mean_df[mean_df['type'] == collaboration_type]['mean_dist_trunc'], mean_df[mean_df['type'] == collaboration_type]['count'])))
        f.write('\n')
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.pearsonr(mean_df[mean_df['type'] == collaboration_type]['mean_dist_trunc'], mean_df[mean_df['type'] == collaboration_type]['count'])))
        f.write('\n')

with open('computer_science/general_analysis/correlation_max_distance_compared_to_citations_by_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['citations'], df[df['type'] == collaboration_type]['distance'])))
        f.write('\n')
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['citations'], df[df['type'] == collaboration_type]['distance'])))
        f.write('\n')

with open('computer_science/general_analysis/correlation_mean_distance_compared_to_citations_by_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['citations'], df[df['type'] == collaboration_type]['mean_distance'])))
        f.write('\n')
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['citations'], df[df['type'] == collaboration_type]['mean_distance'])))
        f.write('\n')

#Plot regression
sns.lmplot(
    x="dist_trunc",
    y="count",
    hue="type",
    data=max_df,
    scatter=False,
)
plt.savefig('computer_science/general_analysis/scatter_max_trunc_distance_count.png')
plt.close()
sns.lmplot(
    x="mean_dist_trunc",
    y="count",
    hue="type",
    data=mean_df,
    scatter=False,
)
plt.savefig('computer_science/general_analysis/scatter_mean_trunc_distance_count.png')
plt.close()

# Test distance by year
means = df.groupby(['year', 'type'])['distance'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="type")
plt.savefig('computer_science/general_analysis/lineplot_max_distance_by_year_by_type.png')
plt.close()
means = df.groupby(['year', 'type'])['mean_distance'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="type")
plt.savefig('computer_science/general_analysis/lineplot_mean_distance_by_year_by_type.png')
plt.close()
# Test distance by year
means = df.groupby(['year', 'type'])['citations'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="type")
plt.savefig('computer_science/general_analysis/lineplot_mean_citations_by_year_by_type.png')
plt.close()

"""
"""#Probability distribution
for collaboration_type in unique_collaboration_types:
    stats.probplot(df[df['type'] == collaboration_type]['distance'], dist="norm", plot=plt)
    plt.title("Probability Plot - " +  collaboration_type)
    plt.show()"""
"""

#Boxplot
df.boxplot(by ='type', column =['distance'], grid = False)
plt.savefig('computer_science/general_analysis/boxplot_max_distance_by_type.png')
plt.close()
df.boxplot(by ='type', column =['mean_distance'], grid = False)
plt.savefig('computer_science/general_analysis/boxplot_mean_distance_by_type.png')
plt.close()

#Denstity
max_df.groupby('type')['dist_trunc'].plot(kind='kde')
plt.legend(['Company', 'Education', 'Mixed'], title='Relationship')
plt.xlabel('Max Distance')
plt.savefig('computer_science/general_analysis/density_max_trunc_distance_by_type.png')
plt.close()
mean_df.groupby('type')['mean_dist_trunc'].plot(kind='kde')
plt.legend(['Company', 'Education', 'Mixed'], title='Relationship')
plt.xlabel('Mean Distance')
plt.savefig('computer_science/general_analysis/density_mean_trunc_distance_by_type.png')
plt.close()

#Probabilty
sns.displot(max_df, x="dist_trunc", hue="type", stat="probability", common_norm=False)
plt.xlabel('Max Distance')
plt.savefig('computer_science/general_analysis/probability_max_trunc_distance_by_type.png')
plt.close()
sns.displot(mean_df, x="mean_dist_trunc", hue="type", stat="probability", common_norm=False)
plt.xlabel('Mean Distance')
plt.savefig('computer_science/general_analysis/probability_mean_trunc_distance_by_type.png')
plt.close()

#Histogram
ax = df.plot.hist(column=["distance"], by="type", figsize=(10, 8))
plt.savefig('computer_science/general_analysis/histogram_max_distance_by_type.png')
plt.close()
ax = df.plot.hist(column=["mean_distance"], by="type", figsize=(10, 8))
plt.savefig('computer_science/general_analysis/histogram_mean_distance_by_type.png')
plt.close()

sns.histplot(
    df, x="distance", y="citations",
    bins=30, pthresh=.05, pmax=.9,
)
plt.savefig('computer_science/general_analysis/histplot_max_distance_compared_to_citations_by_type.png')
plt.close()
sns.histplot(
    df, x="mean_distance", y="citations",
    bins=30, pthresh=.05, pmax=.9,
)
plt.savefig('computer_science/general_analysis/histplot_mean_distance_compared_to_citations_by_type.png')
plt.close()

sns.lmplot(
    x="distance",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.savefig('computer_science/general_analysis/scatter_max_distance_compared_to_citations_by_type.png')
plt.close()
sns.lmplot(
    x="mean_distance",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.savefig('computer_science/general_analysis/scatter_mean_distance_compared_to_citations_by_type.png')
plt.close()

## CONTINENT - COUNTRY ANALYSIS

#region continent
Path("computer_science/continent_analysis/").mkdir(parents=True, exist_ok=True)

df[["type","citations", "international"]].groupby(["type", "international"]).describe().to_csv("computer_science/continent_analysis/describe_citations_by_continent_by_type.csv")
with open('computer_science/continent_analysis/kruskal_citations_by_international_by_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Kruskal Test for citations by type" + str(str(stats.kruskal(df[(df['type'] == collaboration_type) & (df['international'] == True)]['citations'],df[(df['type'] == collaboration_type) & (df['international'] == False)]['citations']))))

for collaboration_type in unique_collaboration_types:
    continent_df = df[df["type"] == collaboration_type]
    new_df = pd.DataFrame({"continent": [], "collaboration": [], "number": int})
    collabs = {"NA": [], "EU": [], "AS": [], "OC": [], "SA": [], "AF": []}
    for index, work in continent_df.iterrows():
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
    plt.savefig(f'computer_science/continent_analysis/pie_continent_collaboration_by_type_{collaboration_type}.png')
    plt.close()

#endregion


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
            dev_type_name = "NO developed countries"
        else:
            dev_type_name = "Developed countries"
        f.write("Spearman test for "+dev_type_name +" - " + str(stats.spearmanr(df_international_max_trunc[df_international_max_trunc['no_dev'] == dev_type]['dist_trunc'], df_international_max_trunc[df_international_max_trunc['no_dev'] == dev_type]['count'])))
        f.write('\n')
        f.write("Pearson test for "+dev_type_name +" - " + str(stats.pearsonr(df_international_max_trunc[df_international_max_trunc['no_dev'] == dev_type]['dist_trunc'], df_international_max_trunc[df_international_max_trunc['no_dev'] == dev_type]['count'])))
        f.write('\n')

df_international_mean_trunc = df_international.groupby(["no_dev", "mean_dist_trunc"]).size().reset_index(name="count")
with open('computer_science/development_analysis/correlation_mean_distance_compared_to_count_by_development.txt', 'w') as f:
    for dev_type in unique_dev_types:
        if dev_type == True:
            dev_type_name = "NO developed countries"
        else:
            dev_type_name = "Developed countries"
        f.write("Spearman test for "+dev_type_name +" - " + str(stats.spearmanr(df_international_mean_trunc[df_international_mean_trunc['no_dev'] == dev_type]['mean_dist_trunc'], df_international_mean_trunc[df_international_mean_trunc['no_dev'] == dev_type]['count'])))
        f.write('\n')
        f.write("Pearson test for "+dev_type_name +" - " + str(stats.pearsonr(df_international_mean_trunc[df_international_mean_trunc['no_dev'] == dev_type]['mean_dist_trunc'], df_international_mean_trunc[df_international_mean_trunc['no_dev'] == dev_type]['count'])))
        f.write('\n')

with open('computer_science/development_analysis/correlation_citations_compared_to_hdi_by_development.txt', 'w') as f:
    for dev_type in unique_dev_types:
        if dev_type == True:
            dev_type_name = "NO developed countries"
        else:
            dev_type_name = "Developed countries"
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
plt.savefig(f'computer_science/development_analysis/scatter_hdi_compared_to_max_distance_by_type.png')
plt.close()
sns.lmplot(
    x="mean_index",
    y="mean_distance",
    hue="type",
    data=df_international,
    scatter=False,
)
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
plt.savefig(f'computer_science/development_analysis/scatter_citations_compared_to_max_distance_by_type.png')
plt.close()
sns.lmplot(
    x="citations",
    y="mean_distance",
    hue="type",
    data=df_international,
    scatter=False,
)
plt.savefig(f'computer_science/development_analysis/scatter_citations_compared_to_mean_distance_by_type.png')
plt.close()

# mean distance by development
df_international.groupby(['no_dev', 'type'])['distance'].mean().unstack().plot(kind='bar')
plt.savefig(f'computer_science/development_analysis/bar_max_distance_by_developement_by_type.png')
plt.close()
df_international.groupby(['no_dev', 'type'])['mean_distance'].mean().unstack().plot(kind='bar')
plt.savefig(f'computer_science/development_analysis/bar_mean_distance_by_developement_by_type.png')
plt.close()

# mean citations by development
df_international.groupby(['no_dev', 'type'])['citations'].mean().unstack().plot(kind='bar')
plt.savefig(f'computer_science/development_analysis/bar_citations_by_developement_by_type.png')
plt.close()

df_international[["type","citations", "no_dev", "distance","mean_distance"]].groupby(["type", "no_dev"]).describe().describe().to_csv("computer_science/development_analysis/describe_citations_and_max_distance_and_mean_distance_by_development_by_type.csv")
no_dev_df = df_international[["type", "citations", "no_dev", "distance"]]
no_dev_df.groupby(["type", "no_dev"])['citations'].count().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.savefig(f'computer_science/development_analysis/pie_citations_count_by_developement_by_type.png')
plt.close()
no_dev_df.groupby(["type", "no_dev"])['citations'].sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.savefig(f'computer_science/development_analysis/pie_citations_sum_by_developement_by_type.png')
plt.close()

with open('computer_science/development_analysis/kruskal_citations_by_developemnt_and_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Kruskal Test for citations by development for " + collaboration_type  + " " + str(stats.kruskal(df_international[(df_international['type'] == collaboration_type) & (df_international['no_dev'] == False)]['citations'], df_international[(df_international['type'] == collaboration_type) & (df_international['no_dev'] == True)]['citations'])))

no_dev_df = df[["citations", "no_dev", "distance", "international"]]
df[["no_dev","citations", "international"]].groupby(["no_dev", "international"]).describe().to_csv("computer_science/development_analysis/describe_citations_by_international_by_development.csv")
no_dev_df.groupby(["no_dev", "international"])['citations'].count().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.savefig(f'computer_science/development_analysis/pie_citations_count_by_developement_by_international.png')
plt.close()
no_dev_df.groupby(["no_dev", "international"])['citations'].sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.savefig(f'computer_science/development_analysis/pie_citations_sum_by_developement_by_international.png')
plt.close()

#endregion

## SCOPE ANALYSIS

Path("computer_science/scope_analysis/").mkdir(parents=True, exist_ok=True)
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

ratio_df = df[df["ratio"] < 1]
ratio_df = ratio_df[ratio_df["ratio"] > 0]

ratio_df['ratio_type'] = ratio_df.apply(f, axis=1)

ratio_df[["ratio_type","citations"]].groupby("ratio_type").describe().to_csv("computer_science/scope_analysis/describe_citations_by_ratio_type.csv")
ratio_df[["ratio_type","distance"]].groupby("ratio_type").describe().to_csv("computer_science/scope_analysis/describe_max_distance_by_ratio_type.csv")
ratio_df[["ratio_type","mean_distance"]].groupby("ratio_type").describe().to_csv("computer_science/scope_analysis/describe_mean_distance_by_ratio_type.csv")

with open('computer_science/scope_analysis/kruskal_max_distance_by_ratio_type.txt', 'w') as f:
    f.write("Kruskal Test for Max distance by ratio type" + str(stats.kruskal(ratio_df[ratio_df['ratio_type'] == "edu"]['distance'], ratio_df[ratio_df['ratio_type'] == "com"]['distance'], ratio_df[ratio_df['ratio_type'] == "half"]['distance'])))
with open('computer_science/scope_analysis/kruskal_mean_distance_by_ratio_type.txt', 'w') as f:
    f.write("Kruskal Test for Mean distance by ratio type" + str(stats.kruskal(ratio_df[ratio_df['ratio_type'] == "edu"]['mean_distance'], ratio_df[ratio_df['ratio_type'] == "com"]['mean_distance'], ratio_df[ratio_df['ratio_type'] == "half"]['mean_distance'])))
with open('computer_science/scope_analysis/kruskal_citations_by_ratio_type.txt', 'w') as f:
    f.write("Kruskal Test for citations by ratio type" + str(stats.kruskal(ratio_df[ratio_df['ratio_type'] == "edu"]['citations'], ratio_df[ratio_df['ratio_type'] == "com"]['citations'], ratio_df[ratio_df['ratio_type'] == "half"]['citations'])))

sns.lmplot(
    x="ratio",
    y="citations",
    data=ratio_df,
    scatter=False,
)
plt.savefig(f'computer_science/scope_analysis/scatter_citations_ratio.png')
plt.close()

ratio_df.groupby(['ratio_type'])['distance'].mean().plot(kind='bar')
plt.savefig(f'computer_science/scope_analysis/bar_max_distance_by_ratio.png')
plt.close()
ratio_df.groupby(['ratio_type'])['mean_distance'].mean().plot(kind='bar')
plt.savefig(f'computer_science/scope_analysis/bar_mean_distance_by_ratio.png')
plt.close()
ratio_df.groupby(['ratio_type'])['citations'].mean().plot(kind='bar')
plt.savefig(f'computer_science/scope_analysis/bar_citations_by_ratio.png')
plt.close()

ratio_df.groupby(["ratio_type"])['citations'].count().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.savefig(f'computer_science/scope_analysis/pie_count_citations_by_ratio.png')
plt.close()
ratio_df.groupby(["ratio_type"])['citations'].sum().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
)
plt.savefig(f'computer_science/scope_analysis/pie_sum_citations_by_ratio.png')
plt.close()
"""


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
unique_collaboration_types = df["type"].unique()
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
