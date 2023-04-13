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
eu_df = pd.read_csv("cs_eu.csv")
unique_collaboration_types = df["type"].unique()
selected_countries=["US","CN","EU"]
## CONTINENT - COUNTRY ANALYSIS

us_collaborations = 0
eu_collaborations = 0
cn_collaborations = 0
us_collaborations_total = 0
eu_collaborations_total = 0
cn_collaborations_total = 0
us_eu_collaborations = 0
us_cn_collaborations = 0
eu_cn_collaborations = 0
eu_cn_us_collaborations = 0
us_citations = 0
eu_citations = 0
cn_citations = 0
us_eu_citations = 0
us_cn_citations = 0
eu_cn_citations = 0
eu_cn_us_citations = 0

for row in tqdm(eu_df.itertuples()):
    locations = literal_eval(row.location)
    country_list= []
    for location in locations:
        country_code = location["country"]
        if country_code in selected_countries:
            country_list.append(country_code)
    citations = row.citations
    country_list = set(country_list)
    if "US" in country_list:
        us_collaborations_total +=1
        if "US" in country_list and len(country_list) == 1:
            us_collaborations += 1
            us_citations += citations
            continue
    if "CN" in country_list:
        cn_collaborations_total +=1
        if "CN" in country_list and len(country_list) == 1:
            cn_collaborations += 1
            cn_citations += citations
            continue
    if "EU" in country_list:
        eu_collaborations_total +=1
        if "EU" in country_list and len(country_list) == 1:
            eu_collaborations += 1
            eu_citations += citations
            continue
    if "EU" in country_list and "CN" in country_list and "US" in country_list:
        eu_cn_us_collaborations += 1
        eu_cn_us_citations += citations
    elif "US" in country_list and "CN" in country_list:
        us_cn_collaborations += 1
        us_cn_citations += citations
    elif "US" in country_list and "EU" in country_list:
        us_eu_collaborations += 1
        us_eu_citations += citations
    elif "EU" in country_list and "CN" in country_list:
        eu_cn_collaborations += 1
        eu_cn_citations += citations
# Define the data
us_data = [us_collaborations, us_eu_collaborations, us_cn_collaborations]
eu_data = [us_eu_collaborations, eu_collaborations, eu_cn_collaborations ]
cn_data = [us_cn_collaborations,eu_cn_collaborations, cn_collaborations]

# Define the x-axis labels
labels = ['US Collaborations', 'EU Collaborations', 'CN Collaborations']

# Define the x-axis locations for each group of bars
x_us = [0, 4, 8]
x_eu = [1, 5, 9]
x_cn = [2, 6, 10]

# Plot the bars
plt.bar(x_us, us_data, color='blue', width=0.8, label='US')
plt.bar(x_eu, eu_data, color='red', width=0.8, label='EU')
plt.bar(x_cn, cn_data, color='green', width=0.8, label='CN')

# Add the x-axis labels and tick marks
plt.xticks([1.5, 5.5, 9.5], labels)
plt.xlabel('Collaboration Type')
plt.ylabel('Number of Collaborations')

# Add a legend
plt.legend()

# Show the plot
plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_cn_us_eu.png')
plt.close()