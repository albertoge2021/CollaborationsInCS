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

us_collaborations = 0
eu_collaborations = 0
cn_collaborations = 0
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

df = pd.read_csv("cs_eu.csv")
countries = ["EU", "US", "CN"]
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    country_list= []
    for location in locations:
        country_code = location["country"]
        if country_code in countries:
            country_list.append(country_code)
    citations = row.citations
    if "US" in set(country_list):
        us_collaborations += 1
        us_citations += citations
        if "CN" in country_list and "EU" in country_list:
            eu_cn_us_collaborations += 1
            eu_cn_us_citations += citations
        if "CN" in country_list:
            us_cn_collaborations += 1
            us_cn_citations += citations
        if "EU" in country_list:
            us_eu_collaborations += 1
            us_eu_citations += citations
    if "CN" in country_list:
        cn_collaborations += 1
        cn_citations += citations
        if "EU" in country_list:
            eu_cn_collaborations += 1
            eu_cn_citations += citations
    if "EU" in country_list:
        eu_collaborations += 1
        eu_citations += citations

us_mean_citations = us_citations / us_collaborations if us_collaborations > 0 else 0
eu_mean_citations = eu_citations / eu_collaborations if eu_collaborations > 0 else 0
cn_mean_citations = cn_citations / cn_collaborations if cn_collaborations > 0 else 0
us_eu_mean_citations = us_eu_citations / us_eu_collaborations if us_eu_collaborations > 0 else 0
us_cn_mean_citations = us_cn_citations / us_cn_collaborations if us_cn_collaborations > 0 else 0
eu_cn_mean_citations = eu_cn_citations / eu_cn_collaborations if eu_cn_collaborations > 0 else 0
eu_cn_us_mean_citations = eu_cn_us_citations / eu_cn_us_collaborations if eu_cn_us_collaborations > 0 else 0

with open("computer_science/country_analysis/country_collaboration_cn_us_eu_citation_mean_total.txt", "w") as f:
    f.write(f"US mean citations: {us_mean_citations}\n")
    f.write(f"EU mean citations: {eu_mean_citations}\n")
    f.write(f"CN mean citations: {cn_mean_citations}\n")
    f.write(f"US-EU mean citations: {us_eu_mean_citations}\n")
    f.write(f"US-CN mean citations: {us_cn_mean_citations}\n")
    f.write(f"EU-CN mean citations: {eu_cn_mean_citations}\n")
    f.write(f"EU-CN-US mean citations: {eu_cn_us_mean_citations}\n")

# Define the data
us_data_means = [us_mean_citations, us_eu_mean_citations, us_cn_mean_citations]
eu_data_means = [us_eu_mean_citations, eu_mean_citations, eu_cn_mean_citations]
cn_data_means = [eu_cn_mean_citations, us_cn_mean_citations, cn_mean_citations]

# Define the x-axis labels
labels = ['US Collaborations', 'EU Collaborations', 'CN Collaborations']

# Define the x-axis locations for each group of bars
x_us = [0, 4, 8]
x_eu = [1, 5, 9]
x_cn = [2, 6, 10]

# Plot the bars
plt.bar(x_us, us_data_means, color='blue', width=0.8, label='US')
plt.bar(x_eu, eu_data_means, color='red', width=0.8, label='EU')
plt.bar(x_cn, cn_data_means, color='green', width=0.8, label='CN')

# Add the x-axis labels and tick marks
plt.xticks([1.5, 5.5, 9.5], labels)
plt.xlabel('Collaboration Type')
plt.ylabel('Number of Collaborations')

# Add a legend
plt.legend()

# Show the plot
plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_citations_cn_us_eu_total.png')
plt.close()