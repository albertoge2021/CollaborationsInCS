from ast import literal_eval
from collections import Counter
import pandas as pd
import warnings
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
from pathlib import Path
from collections import defaultdict


warnings.simplefilter(action="ignore", category=FutureWarning)


eu_df = pd.read_csv("cs_eu.csv")
selected_countries=["US","CN","EU"]
unique_collaboration_types = eu_df["type"].unique()

us_ratio_total = []
eu_ratio_total = []
cn_ratio_total = []
us_eu_counts = 0
us_cn_counts = 0
eu_cn_counts = 0
us_citations = 0
eu_citations = 0
cn_citations = 0

for row in tqdm(eu_df.itertuples()):
    us_counts = 0
    eu_counts = 0
    cn_counts = 0
    locations = literal_eval(row.location)
    country_list= []
    for location in locations:
        country_list.append(location["country"])
    if any(country in selected_countries for country in country_list):
        num_countries = len(country_list)
        citations = row.citations
        if "US" in country_list:
            us_counts += country_list.count("US")
            us_citations += citations
        if "CN" in country_list:
            cn_counts += country_list.count("CN")
            cn_citations += citations
        if "EU" in country_list:
            eu_counts += country_list.count("EU")
            eu_citations += citations
        if us_counts > 0:
            us_ratio_total.append(((us_counts/num_countries), citations, row.type))
        if eu_counts > 0:
            eu_ratio_total.append(((eu_counts/num_countries), citations, row.type))
        if cn_counts > 0:
            cn_ratio_total.append(((cn_counts/num_countries), citations, row.type))

df = pd.DataFrame(us_ratio_total, columns=['ratio', 'citations', 'type'])

with open('computer_science/country_analysis/correlation_ratio_compared_to_citations_by_type_us.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
    f.write("Pearson test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')
    f.write("Spearman test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')

sns.lmplot(
    x="ratio",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('US participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_by_type_us.png')
plt.close()

sns.lmplot(
    x="ratio",
    y="citations",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('US participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_us.png')
plt.close()

means = df.groupby(['ratio',"type"])['citations'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="ratio", y="mean", hue="type")
plt.xlabel('Participation ratio')
plt.ylabel('Mean citations')
plt.title('US participation ratio vs mean citations')
plt.savefig(f'computer_science/country_analysis/scatter_mean_citations_by_ratio_by_type_us.png')
plt.close()

df = pd.DataFrame(eu_ratio_total, columns=['ratio', 'citations', 'type'])

with open('computer_science/country_analysis/correlation_ratio_compared_to_citations_by_type_eu.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
    f.write("Pearson test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')
    f.write("Spearman test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')

sns.lmplot(
    x="ratio",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('EU participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_by_type_eu.png')
plt.close()

sns.lmplot(
    x="ratio",
    y="citations",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('EU participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_eu.png')
plt.close()

means = df.groupby(['ratio', "type"])['citations'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="ratio", y="mean", hue="type")
plt.xlabel('Participation ratio')
plt.ylabel('Mean citations')
plt.title('EU participation ratio vs mean citations')
plt.savefig(f'computer_science/country_analysis/scatter_mean_citations_by_ratio_by_type_eu.png')
plt.close()

df = pd.DataFrame(cn_ratio_total, columns=['ratio', 'citations', 'type'])

with open('computer_science/country_analysis/correlation_ratio_compared_to_citations_by_type_cn.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
    f.write("Pearson test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')
    f.write("Spearman test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')

sns.lmplot(
    x="ratio",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('CN participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_by_type_cn.png')
plt.close()

sns.lmplot(
    x="ratio",
    y="citations",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('CN participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_cn.png')
plt.close()

means = df.groupby(['ratio',"type"])['citations'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="ratio", y="mean", hue="type")
plt.xlabel('Participation ratio')
plt.ylabel('Mean citations')
plt.title('CN participation ratio vs mean citations')
plt.savefig(f'computer_science/country_analysis/scatter_mean_citations_by_ratio_by_type_cn.png')
plt.close()
    