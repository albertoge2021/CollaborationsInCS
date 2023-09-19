from ast import literal_eval
from collections import Counter, defaultdict
import csv
from matplotlib import patches
import pandas as pd
import warnings
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pycountry
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import pycountry_convert as pc
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

# Setup Data
df = pd.read_csv("cs_dataset_final.csv")
eng_df = pd.read_csv("english_level.csv")
df = df[df["year"] > 1989]
df = df[df["year"] < 2022]
df = df.drop_duplicates()
df = df.dropna()
df["num_items"] = df["countries"].apply(lambda x: len(x))
df_filtered = df[df["num_items"] >= 2]
df = df_filtered.drop("num_items", axis=1)

unique_collaboration_types = df["type"].unique()
selected_countries = ["US", "CN", "EU"]
colors = ["deepskyblue", "limegreen", "orangered", "mediumpurple"]
Path("paper_results_2/").mkdir(parents=True, exist_ok=True)
EU_COUNTRIES = [
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "GB",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
]

eu_countries_alpha_3 = [
    "AUT",
    "BEL",
    "BGR",
    "HRV",
    "CYP",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "IRL",
    "ITA",
    "LVA",
    "LTU",
    "LUX",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "ESP",
    "SWE",
    "GBR",
]

# Remove rows with specific collaborators
collaborators_to_remove = [
    "USA",
    "CHN",
    "AUT",
    "BEL",
    "BGR",
    "HRV",
    "CYP",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "IRL",
    "ITA",
    "LVA",
    "LTU",
    "LUX",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "ESP",
    "SWE",
    "GBR",
]

dev_df = pd.read_csv("human_dev_standard.csv")

"""dev_df = pd.read_csv("human_dev_standard.csv")
def get_iso_code(country_name):
    try:
        country_code = pc.country_name_to_country_alpha2(country_name, cn_name_format="default")
        return country_code
    except Exception:
        return None

# Apply the function to the "Country" column
dev_df["Code"] = dev_df["Code"].apply(get_iso_code)
dev_df.to_csv("human_dev_standard.csv", index=False)

hdi_dict = {}

# Populate the hdi_dict
for row in dev_df.itertuples():
    year = row.Year
    code = row.Code
    hdi = row.Hdi
    if year not in hdi_dict:
        hdi_dict[year] = {}
    hdi_dict[year][code] = hdi

# Create a function to check if a year and country meet the condition
def has_lower_hdi(year, country_list):
    # Check if the year is in hdi_dict
    if year in hdi_dict:
        # Iterate through the country_list
        for country_code in country_list:
            # Check if there is a country with lower HDI
            if country_code in hdi_dict[year] and hdi_dict[year][country_code] < 0.549:
                return True
    return False

dev_list = []
for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    country_list = literal_eval(row.countries)
    country_list = ["RS" if country == "XK" else "CN" if country == "TW" else country for country in country_list]
    country_list = list(set(country_list))
    year = row.year
    # Check if the condition is met for the current row
    condition_met = has_lower_hdi(year, country_list)
    has_no_dev_country = False
    if condition_met:
        has_no_dev_country = True
    dev_list.append((
        row.citations,
        row.year,
        row.concepts,
        row.type,
        row.countries,
        row.max_distance,
        row.avg_distance,
        has_no_dev_country)
    )

dev_df_gropued = pd.DataFrame(dev_list, columns=['citations', 'year', 'concepts', 'type', 'countries', 'max_distance', 'avg_distance', 'has_no_dev_country'])
print(dev_df_gropued.groupby('has_no_dev_country').count())
dev_df_gropued.to_csv("dev_df_gropued.csv", index=False)
"""


total_collaborations = defaultdict(dict)

for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    country_list = literal_eval(row.countries)
    country_list = ["RS" if country == "XK" else "CN" if country == "TW" else country for country in country_list]
    country_list = list(set(country_list))

    # Iterate through each combination of two countries in the current row
    for i in range(len(country_list)):
        for j in range(i + 1, len(country_list)):
            country1, country2 = country_list[i], country_list[j]
            # Exclude self-collaborations and handle None values
            if country1 is not None and country2 is not None and country1 != country2:
                # Update total_collaborations for country1
                total_collaborations[country1][country2] = total_collaborations[country1].get(country2, 0) + 1

                # Update total_collaborations for country2
                total_collaborations[country2][country1] = total_collaborations[country2].get(country1, 0) + 1

# Sort the total_collaborations dictionary by keys and create a new sorted dictionary
sorted_collaborations = {country: dict(sorted(collabs.items())) for country, collabs in total_collaborations.items()}
total_collaborations_by_country = defaultdict(int)

# Iterate through the sorted_collaborations dictionary and accumulate the counts
for country, collaborations in sorted_collaborations.items():
    total_collaborations = sum(collaborations.values())
    total_collaborations_by_country[country] = total_collaborations

# Convert the defaultdict to a regular dictionary if needed
total_collaborations_by_country = dict(total_collaborations_by_country)
collaborations_df = pd.DataFrame(total_collaborations_by_country.items(), columns=["Country", "Collaborations"])

# Merge 'eng_df' and 'collaborations_df' by the "Country" column
merged_df = pd.merge(eng_df[["Country", "Score"]], collaborations_df, on="Country", how="left")

# Drop rows with missing or infinite values
merged_df = merged_df.dropna()
merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()

# Create a scatter plot to visualize the relationship between Score and Collaborations
plt.figure(figsize=(10, 6))
sns.lmplot(
    x="Score",
    y="Collaborations",
    data=merged_df,
)
plt.title("Correlation Plot between Score and Collaborations")
plt.xlabel("Score")
plt.ylabel("Collaborations")
plt.grid(True)
plt.savefig(f"paper_results_2/correlation_number_papers_english_level.png")
plt.close()

corr_coeff, p_value = pearsonr(merged_df["Score"], merged_df["Collaborations"])
corr_coeff_spear, p_value_spear = spearmanr(merged_df["Score"], merged_df["Collaborations"])
with open('paper_results_2/enlgish_level_results.txt', 'w') as file:
    file.write(f"Pearson Correlation Coefficient: {corr_coeff:.2f}" + " - P-Value: {p_value:.5f}")
    file.write(f"Pearson Correlation Coefficient: {corr_coeff_spear:.2f}" + " - P-Value: {p_value_spear:.5f}")

collaborations_by_year_and_country = defaultdict(lambda: defaultdict(dict))

# Iterate through the data and accumulate collaborations by year, country, and other country
for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    year = row.year
    country_list = literal_eval(row.countries)
    country_list = ["RS" if country == "XK" else "CN" if country == "TW" else country for country in country_list]
    country_list = list(set(country_list))

    # Iterate through each combination of two countries in the current row
    for i in range(len(country_list)):
        for j in range(i + 1, len(country_list)):
            country1, country2 = country_list[i], country_list[j]
            
            # Exclude self-collaborations and handle None values
            if country1 is not None and country2 is not None and country1 != country2:
                # Update collaborations_by_year_and_country for country1
                collaborations_by_year_and_country[year][country1][country2] = collaborations_by_year_and_country[year][country1].get(country2, 0) + 1

                # Update collaborations_by_year_and_country for country2
                collaborations_by_year_and_country[year][country2][country1] = collaborations_by_year_and_country[year][country2].get(country1, 0) + 1

# Convert the defaultdict to a regular dictionary if needed
collaborations_by_year_and_country = {year: {country: dict(data) for country, data in data_dict.items()} for year, data_dict in collaborations_by_year_and_country.items()}
total_collaborations_by_year_and_country = defaultdict(lambda: defaultdict(int))

# Iterate through collaborations_by_year_and_country to calculate total collaborations
for year, year_data in collaborations_by_year_and_country.items():
    for country1, country_data in year_data.items():
        total_collaborations = sum(country_data.values())
        total_collaborations_by_year_and_country[year][country1] = total_collaborations

# Convert the defaultdict to a regular dictionary if needed
total_collaborations_by_year_and_country = {year: dict(data) for year, data in total_collaborations_by_year_and_country.items()}

data_list = []
for year, year_data in total_collaborations_by_year_and_country.items():
    for country, total_collaborations in year_data.items():
        data_list.append({
            'Year': year,
            'Country': country,
            'TotalCollaborations': total_collaborations
        })

# Create a DataFrame from the list of dictionaries
total_collaborations_df = pd.DataFrame(data_list)

merged_data = pd.merge(total_collaborations_df, dev_df, left_on=['Year', 'Country'], right_on=['Year', 'Code'], how='left')

# Drop rows with missing values in the 'Hdi' column
merged_data = merged_data.dropna(subset=['Hdi'])

# 4. HDI Grouping
# Create HDI groups
bins = [0, 0.549, 0.699, 0.799, 1.0]
labels = ['Low', 'Medium', 'High', 'Very High"']
merged_data['HdiGroup'] = pd.cut(merged_data['Hdi'], bins=bins, labels=labels)

# Calculate average total collaborations by HDI group
average_collaborations_by_hdi = merged_data.groupby('HdiGroup')['TotalCollaborations'].mean()
with open('paper_results_2/hdi_results.txt', 'a') as file:
    file.write("Average Total Collaborations by HDI Group:\n")
    file.write(average_collaborations_by_hdi.to_string())
    file.write("\n")

corr_coeff, p_value = pearsonr(merged_data["Hdi"], merged_data["TotalCollaborations"])
corr_coeff_spear, p_value_spear = spearmanr(merged_data["Hdi"], merged_data["TotalCollaborations"])
with open('paper_results_2/hdi_results.txt', 'a') as file:
    file.write(f"Pearson Correlation Coefficient: {corr_coeff:.2f}" + f" - P-Value: {p_value:.5f}\n")
    file.write(f"Spearman Correlation Coefficient: {corr_coeff_spear:.2f}" + f" - P-Value: {p_value_spear:.5f}\n")

"""dev_df_gropued = pd.read_csv("dev_df_gropued.csv")
hdi_dict = {}

# Populate the hdi_dict
for row in dev_df.itertuples():
    year = row.Year
    code = row.Code
    hdi = row.Hdi
    if year not in hdi_dict:
        hdi_dict[year] = {}
    hdi_dict[year][code] = hdi"""
# Plot topics by group 
# Plot topics by group by year
# average citations
# correlation between the citations and average hdi
# pie chart for topics