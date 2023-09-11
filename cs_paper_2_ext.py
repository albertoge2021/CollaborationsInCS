from ast import literal_eval
from collections import Counter
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


warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

# Setup Data
df = pd.read_csv("cs_dataset_final.csv")
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


# TODO For each region, which countries are the most common collaborators? Pie/Bar chart
# TODO Do these 3 regions collaborate more with each other than with other regions?
# TODO When they collaborate, in which proportion is done? And how it affects the number of citations?
# TODO Which other regions do the others collaborate with?
# TODO Citations rates of both groups, also by type
# TODO Average distance between the collaborations

#region preliminary_analysis

countries_ratio = []
countries_df = []
countries_to_remove = ["US", "EU", "CN"]

for row in tqdm(df.itertuples()):
    country_list = set(literal_eval(row.countries))
    if len(set(country_list)) < 1:
        continue
    updated_list = []
    for country in country_list:
        country = "RS" if country == "XK" else country
        country = "CN" if country == "TW" else country
        country = "EU" if country in EU_COUNTRIES else country
        updated_list.append(country)
    citations = int(row.citations)
    if all(code in updated_list for code in ["EU", "CN", "US"]):
        continue
    if any(code in updated_list for code in ["EU", "CN", "US"]):
        relation =  "EU-US-CN"
    else:
        relation = "Other countries"
    countries_ratio.append(
            (
                citations,
                row.type,
                row.year,
                relation,
                row.max_distance,
                row.avg_distance,
            )
        )
    
    updated_list = [country for country in updated_list if country not in countries_to_remove]
    for country in updated_list:
        countries_df.append(
                (
                    citations,
                    row.type,
                    row.year,
                    relation,
                    country,
                    row.max_distance,
                    row.avg_distance,
                )
            )

countries_df = pd.DataFrame(
    countries_df,
    columns=[
        "citations",
        "type",
        "year",
        "relation",
        "country",
        "max_distance",
        "avg_distance",
    ],
)

ratio_df = pd.DataFrame(
    countries_ratio,
    columns=[
        "citations",
        "type",
        "year",
        "relation",
        "max_distance",
        "avg_distance",
    ],
)

relation_country_occurrences = countries_df.groupby(['relation', 'country']).size().reset_index(name='occurrences')
relation_country_occurrences = relation_country_occurrences.sort_values(by=['relation', 'occurrences'], ascending=[True, False])
unique_relations = relation_country_occurrences['relation'].unique()

for relation in unique_relations:
    relation_data = relation_country_occurrences[relation_country_occurrences['relation'] == relation].head(10)
    
    plt.figure(figsize=(10, 6))
    plt.bar(relation_data['country'], relation_data['occurrences'])
    plt.xlabel('Country')
    plt.ylabel('Number of Occurrences')
    plt.title(f'Top 10 Most Repeated Countries for Relation: {relation}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"paper_results_2/barplot_most_collaborators_{relation}.png")
    plt.close()

means = ratio_df.groupby(["relation", "year"]).size().reset_index(name="count")
sns.lineplot(data=means, x="year", y="count", hue="relation")
plt.xlabel("Year")
plt.ylabel("Number of publications")
plt.title("Number of publications by regions per year")
plt.tight_layout()
plt.savefig(f"paper_results_2/lineplot_collaborations_per_year_per_country.png")
plt.close()

relation_counts = ratio_df['relation'].value_counts()
relation_counts.plot(kind='bar')
plt.title('Number of papers per Group of Countries')
plt.xlabel('Group of Countries')
plt.ylabel('Number of papers')
plt.tight_layout()
plt.savefig(f"paper_results_2/barplot_collaborations_per_country_group.png")
plt.close()

relation_counts = ratio_df.groupby(['type'])['relation'].value_counts()
relation_counts.plot(kind='bar')
plt.title('Number of papers per Group of Countries per type')
plt.xlabel('Group of Countries')
plt.ylabel('Number of papers')
plt.legend(title='Type')
plt.tight_layout()
plt.savefig(f"paper_results_2/barplot_publications_per_country_group_per_type.png")
plt.close()

avg_citations_per_country_group = ratio_df.groupby('relation')['citations'].mean()
avg_citations_per_country_group.plot(kind='bar')
plt.title('Average Citations per Group of Countries')
plt.xlabel('Group of Countries')
plt.ylabel('Average Citations')
plt.tight_layout()
plt.savefig(f"paper_results_2/barplot_avg_citations_per_country_group.png")
plt.close()

max_distance_per_country_group = ratio_df.groupby('relation')['max_distance'].mean()
max_distance_per_country_group.plot(kind='bar')
plt.title('Average Max Distance per Group of Countries')
plt.xlabel('Group of Countries')
plt.ylabel('Average Max Distance')
plt.tight_layout()
plt.savefig(f"paper_results_2/barplot_max_distance_per_country_group.png")
plt.close()

avg_distance_per_country_group = ratio_df.groupby('relation')['avg_distance'].mean()
avg_distance_per_country_group.plot(kind='bar')
plt.title('Average Mean Distance per Group of Countries')
plt.xlabel('Group of Countries')
plt.ylabel('Average Mean Distance')
plt.tight_layout()
plt.savefig(f"paper_results_2/barplot_avg_distance_per_country_group.png")
plt.close()

avg_citations_per_type_relation = ratio_df.groupby(['type', 'relation'])['citations'].mean().reset_index()
sns.barplot(x='relation', y='citations', hue='type', data=avg_citations_per_type_relation)
plt.title('Average Citations per Group of Countries by Type')
plt.xlabel('Group of Countries')
plt.ylabel('Average Citations')
plt.legend(title='Type')
plt.tight_layout()
plt.savefig(f"paper_results_2/barplot_avg_distance_per_country_group_per_type.png")
plt.close()

countries_ratio = []

for row in tqdm(df.itertuples()):
    country_list = set(literal_eval(row.countries))
    if len(set(country_list)) < 1:
        continue
    updated_list = []
    for country in country_list:
        country = "RS" if country == "XK" else country
        country = "CN" if country == "TW" else country
        country = "EU" if country in EU_COUNTRIES else country
        updated_list.append(country)
    citations = int(row.citations)
    if all(code in updated_list for code in ["EU", "CN", "US"]):
        continue
    elif any(code in updated_list for code in ["EU", "CN", "US"]):
        relation =  "-".join(code for code in ["EU", "CN", "US"] if code in updated_list)
    else:
        relation = "Other countries"
    countries_ratio.append(
            (
                citations,
                row.type,
                row.year,
                relation,
                row.max_distance,
                row.avg_distance,
            )
        )

ratio_df = pd.DataFrame(
    countries_ratio,
    columns=[
        "citations",
        "type",
        "year",
        "relation",
        "max_distance",
        "avg_distance",
    ],
)

means = ratio_df.groupby(["relation", "year"]).size().reset_index(name="count")
sns.lineplot(data=means, x="year", y="count", hue="relation")
plt.xlabel("Year")
plt.ylabel("Number of collaborations")
plt.title("Number of collaborations by regions per year")
plt.savefig(f"paper_results_2/lineplot_collaborations_per_year_per_country_group.png")
plt.close()

countries_ratio = []

for row in tqdm(df.itertuples()):
    country_list = set(literal_eval(row.countries))
    if len(set(country_list)) < 1:
        continue
    updated_list = []
    for country in country_list:
        country = "RS" if country == "XK" else country
        country = "CN" if country == "TW" else country
        country = "EU" if country in EU_COUNTRIES else country
        updated_list.append(country)
    citations = int(row.citations)
    for country in updated_list:
        countries_ratio.append(
                (
                    citations,
                    row.type,
                    row.year,
                    country,
                    row.max_distance,
                    row.avg_distance,
                )
            )

ratio_df = pd.DataFrame(
    countries_ratio,
    columns=[
        "citations",
        "type",
        "year",
        "country",
        "max_distance",
        "avg_distance",
    ],
)

value_counts = (
    ratio_df["country"]
    .value_counts()
    .rename_axis("country")
    .reset_index(name="count")
)

# Sort the DataFrame by count in descending order
value_counts = value_counts.sort_values("count", ascending=False)

# Plotting the bar plot for the top 15 values
top_values = value_counts.head(10)
top_values.plot(kind="bar", x="country", y="count", figsize=(10, 6))
plt.xlabel("Country")
plt.ylabel("Publications")
plt.title("Top 10 Values with the Most Publications")
plt.savefig(f"paper_results_2/bar_collaborations_by_country_top_10.png")
plt.close()

country_counts = (
    ratio_df.groupby(["year", "country"]).size().reset_index(name="count")
)

# Plotting the bar plot for the top 15 values
country_year_counts = value_counts[~value_counts['country'].isin(selected_countries)]
top_values = country_year_counts.head(10)
top_values.plot(kind="bar", x="country", y="count", figsize=(10, 6))
plt.xlabel("Country")
plt.ylabel("Publications")
plt.title("Top 10 Values with the Most Publications (Without CN, EU, US)")
plt.savefig(f"paper_results_2/bar_collaborations_by_country_top_10_without_eu_cn_us.png")
plt.close()

country_counts = (
    ratio_df.groupby(["year", "country"]).size().reset_index(name="count")
)

# Counting the number of occurrences per year per country
country_year_counts = ratio_df.groupby(['year', 'country']).size().reset_index(name='count')

# Excluding specific countries (CN, EU, US)
excluded_countries = ['CN', 'EU', 'US']
country_year_counts = country_year_counts[~country_year_counts['country'].isin(excluded_countries)]

# Finding the top 10 countries overall in time
top_countries = country_year_counts.groupby('country')['count'].sum().nlargest(10).index

# Filtering the data to include only the top 10 countries
top_countries_data = country_year_counts[country_year_counts['country'].isin(top_countries)]

# Plotting the data using seaborn
sns.lineplot(x='year', y='count', hue='country', data=top_countries_data)
plt.title('Number of Occurrences per Year per Country (Top 10, Excluding CN, EU, US)')
plt.xlabel('Year')
plt.ylabel('Number of Occurrences')
plt.legend(title='Country')
plt.tight_layout()
plt.savefig(f"paper_results_2/lineplot_papers_per_year_per_country_without_eu_us_cn.png")
plt.close()

# Grouping the data by 'country' and calculating the average of 'citations'
country_avg_citations = ratio_df.groupby('country')['citations'].mean().reset_index()

# Finding the top 10 countries with highest average citations
top_avg_citations_countries = country_avg_citations.nlargest(10, 'citations')

# Plotting the data using seaborn
sns.barplot(y='citations', x='country', data=top_avg_citations_countries)
plt.title('Top 10 Countries with Highest Average Citations')
plt.xlabel('Average Citations')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig(f"paper_results_2/barplot_citation_per_country.png")
plt.close()

# Grouping the data by 'country' and calculating the average of 'citations'
country_avg_citations = ratio_df.groupby('country')['citations'].mean().reset_index()

# Excluding specific countries (CN, EU, US)
excluded_countries = ['CN', 'EU', 'US']
country_avg_citations = country_avg_citations[~country_avg_citations['country'].isin(excluded_countries)]

# Finding the top 10 countries with highest average citations
top_avg_citations_countries = country_avg_citations.nlargest(10, 'citations')

# Plotting the data using seaborn
sns.barplot(y='citations', x='country', data=top_avg_citations_countries)
plt.title('Top 10 Countries with Highest Average Citations (Excluding CN, EU, US)')
plt.xlabel('Average Citations')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig(f"paper_results_2/barplot_citation_per_country_without_eu_us_cn.png")
plt.close()

countries_ratio = []

for row in tqdm(df.itertuples()):
    country_list = literal_eval(row.countries)
    if len(set(country_list)) < 1:
        continue
    for country in country_list:
        country = "RS" if country == "XK" else country
        country = "CN" if country == "TW" else country
        country = "EU" if country in EU_COUNTRIES else country
        countries_ratio.append(country)

#TODO porcentage de colaboracion nacional entre los dos grupos
#TODO porcentage de colaboracion grupal entre los dos grupos
#TODO porcentage de colaboracion internacional entre los dos grupos

#endregion

#region maps

collaborators = []

for row in tqdm(df.itertuples()):
    country_list = set(literal_eval(row.countries))
    if len(set(country_list)) < 1:
        continue
    if len(country_list) < 1:
        continue
    updated_list = []
    for country in country_list:
        if country is None:
            continue
        country = "RS" if country == "XK" else country
        country = "CN" if country == "TW" else country
        updated_list.append(pycountry.countries.get(alpha_2=country).alpha_3)
    if (
        any(country_code in updated_list for country_code in eu_countries_alpha_3)
        and len(updated_list) > 1
    ):
        for country in updated_list:
            collaborators.append(
                (
                    "EU",
                    country,
                )
            )
            # collaborators.append(("EU", country, int(row.citations),row.type))
    if "USA" in updated_list and len(updated_list) > 1:
        for country in updated_list:
            collaborators.append(
                (
                    "USA",
                    country,
                )
            )
            # collaborators.append(("USA", country, int(row.citations),row.type))
    if "CHN" in updated_list and len(updated_list) > 1:
        for country in updated_list:
            collaborators.append(
                (
                    "CHN",
                    country,
                )
            )
            # collaborators.append(("CHN", country, int(row.citations),row.type))

# participation_df = pd.DataFrame(collaborators, columns=["origin", "collaborator", "citations", "type"])
participation_df = pd.DataFrame(collaborators, columns=["origin", "collaborator"])
participation_df = participation_df[
    ~participation_df["collaborator"].isin(collaborators_to_remove)
]
world_map = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
grouped_df = (
    participation_df.groupby(["origin", "collaborator"])
    .size()
    .reset_index(name="frequency")
)

for origin in participation_df["origin"].unique():
    subset_df = grouped_df[grouped_df["origin"] == origin]
    merged_df = world_map.merge(
        subset_df, left_on="iso_a3", right_on="collaborator", how="left"
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    merged_df.plot(
        column="frequency",
        cmap="YlOrRd",
        linewidth=0.8,
        ax=ax,
        edgecolor="0.8",
        legend=True,
    )
    merged_df[merged_df["iso_a3"].isin(collaborators_to_remove)].plot(
        color="mediumpurple", ax=ax
    )
    if origin == "EU":
        origin_name = "the European Union"
    elif origin == "USA":
        origin_name = "the United States of America"
    elif origin == "CHN":
        origin_name = "China"
    ax.set_title(f"Number of collaborations with {origin_name}")
    plt.savefig(
        f"paper_results_2/map_number_of_collaborations_{origin}.png"
    )
    plt.close()

merged_df = world_map.merge(
    grouped_df, left_on="iso_a3", right_on="collaborator", how="left"
)
fig, ax = plt.subplots(figsize=(12, 8))
merged_df.plot(
    column="frequency",
    cmap="YlOrRd",
    linewidth=0.8,
    ax=ax,
    edgecolor="0.8",
    legend=True,
)
merged_df[merged_df["iso_a3"].isin(collaborators_to_remove)].plot(
    color="mediumpurple", ax=ax
)
ax.set_title(f"Number of collaborations")
plt.savefig(
    f"paper_results_2/map_number_of_collaborations.png"
)
plt.close()

grouped_df = (
    participation_df.groupby(["origin", "collaborator"])
    .size()
    .reset_index(name="frequency")
)
total_occurrences = grouped_df.groupby("origin")["frequency"].sum()
for origin in participation_df["origin"].unique():
    subset_df = grouped_df[grouped_df["origin"] == origin]
    subset_df["percentage"] = subset_df["frequency"] / total_occurrences[origin] * 100
    merged_df = world_map.merge(
        subset_df, left_on="iso_a3", right_on="collaborator", how="left"
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    merged_df.plot(
        column="percentage",
        cmap="YlOrRd",
        linewidth=0.8,
        ax=ax,
        edgecolor="0.8",
        legend=True,
        vmin=0, vmax=40
    )
    merged_df[merged_df["iso_a3"].isin(collaborators_to_remove)].plot(
        color="mediumpurple", ax=ax
    )
    if origin == "EU":
        origin_name = "the European Union"
    elif origin == "USA":
        origin_name = "the United States of America"
    elif origin == "CHN":
        origin_name = "China"
        legend_label = "Percentage of Collaborations"
    ax.set_title(f"Percentage of collaborations with {origin_name}")
    plt.savefig(
        f"paper_results_2/map_percentage_of_collaborations_{origin}.png"
    )
    plt.close()

grouped_df["percentage"] = grouped_df["frequency"] / total_occurrences[origin] * 100
merged_df = world_map.merge(
    grouped_df, left_on="iso_a3", right_on="collaborator", how="left"
)
fig, ax = plt.subplots(figsize=(12, 8))
merged_df.plot(
    column="percentage",
    cmap="YlOrRd",
    linewidth=0.8,
    ax=ax,
    edgecolor="0.8",
    legend=True,
    vmin=0, vmax=40
)
merged_df[merged_df["iso_a3"].isin(collaborators_to_remove)].plot(
    color="mediumpurple", ax=ax
)
ax.set_title(f"Percentage of collaborations")
plt.savefig(
    f"paper_results_2/map_percentage_of_collaborations.png"
)
plt.close()

#endregion

new_df = []
dev_df = pd.read_csv("human_dev_standard.csv")
result_dict = {}

# Iterate through the DataFrame rows
for row in dev_df.itertuples():
    code_2 = row.Code_2
    year = row.Year
    index = row.Hdi
    
    # If the code_2 key doesn't exist in the dictionary, create it
    if code_2 not in result_dict:
        result_dict[code_2] = {}
    
    # Add the year and index to the inner dictionary
    result_dict[code_2][year] = index

# Create a mapping for code replacement
country_mapping = {
    "XK": "RS",
    **{country: "EU" for country in EU_COUNTRIES}
}

for row in tqdm(df.itertuples()):
    country_list = set(literal_eval(row.countries))
    if len(set(country_list)) < 1:
        continue
    
    # Replace codes and create an updated country list
    updated_list = [country_mapping.get(country, country) for country in country_list]
    
    # Check if all target codes are in the updated list
    if set(selected_countries).issubset(updated_list):
        continue
    
    citations = int(row.citations)
    
    for country in updated_list:
        hdi = result_dict.get(country, {}).get(row.year, None)  # Get HDI from the dictionary
        if any(code == "EU" for code in updated_list):
            new_df.append(
                (
                    citations,
                    row.type,
                    row.year,
                    "EU",
                    country,
                    row.max_distance,
                    row.avg_distance,
                    hdi
                )
            )
        if any(code == "US" for code in updated_list):
            new_df.append(
                (
                    citations,
                    row.type,
                    row.year,
                    "US",
                    country,
                    row.max_distance,
                    row.avg_distance,
                    hdi
                )
            )
        if any(code == "CN" for code in updated_list):
            new_df.append(
                (
                    citations,
                    row.type,
                    row.year,
                    "CN",
                    country,
                    row.max_distance,
                    row.avg_distance,
                    hdi
                )
            )

new_df = pd.DataFrame(
    new_df,
    columns=[
        "citations",
        "type",
        "year",
        "relation",
        "country",
        "max_distance",
        "avg_distance",
        "hdi"
    ],
)

new_df = new_df[~new_df['country'].isin(selected_countries)]

# Group by 'relation' and 'country', calculate mean citations, and sort
grouped_citations = new_df.groupby(['relation', 'country'])['citations'].mean().reset_index()
grouped_citations = grouped_citations.sort_values(by=['relation', 'citations'], ascending=[True, False])

# Group by 'relation' and 'country', calculate the count of rows, and sort
grouped_collaborations = new_df.groupby(['relation', 'country']).size().reset_index(name='row_count')
grouped_collaborations = grouped_collaborations.sort_values(by=['relation', 'row_count'], ascending=[True, False])

# Group by 'relation' and 'country', calculate the highest avg_distance, and sort
grouped_avg_distance = new_df.groupby(['relation', 'country'])['avg_distance'].max().reset_index()
grouped_avg_distance = grouped_avg_distance.sort_values(by=['relation', 'avg_distance'], ascending=[True, False])

# Group by 'relation' and 'country', calculate the highest max_distance, and sort
grouped_max_distance = new_df.groupby(['relation', 'country'])['max_distance'].max().reset_index()
grouped_max_distance = grouped_max_distance.sort_values(by=['relation', 'max_distance'], ascending=[True, False])

unique_relations = grouped_citations['relation'].unique()

sns.lmplot(
    x="hdi",
    y="citations",
    data=new_df,
    scatter=False,
)
plt.xlabel('HDI')
plt.ylabel('Citations')
plt.title('Relationship between HDI and Citations')
plt.tight_layout()
plt.savefig('paper_results_2/scatter_general_hdi_citations.png')
plt.close()

for relation in unique_relations:
    # Filter data for the current relation based on top_countries
    top_countries_data = grouped_citations[
        (grouped_citations['relation'] == relation)
    ].head(10)
    
    # Citations
    plt.figure(figsize=(10, 6))
    plt.bar(top_countries_data['country'], top_countries_data['citations'])
    plt.xlabel('Country')
    plt.ylabel('Average Citations')
    plt.title(f'Top 10 Countries with Highest Avg Citations for Relation: {relation}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"paper_results_2/barplot_avg_citations_by_country_{relation}.png")
    plt.close()

    # Collaborations
    top_countries_data = grouped_collaborations[
        (grouped_collaborations['relation'] == relation)
    ].head(10)
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_countries_data['country'], top_countries_data['row_count'])
    plt.xlabel('Country')
    plt.ylabel('Total Collaborations')
    plt.title(f'Top 10 Countries by Number of Collaborations for Relation: {relation}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"paper_results_2/barplot_collaborations_by_country_{relation}.png")
    plt.close()

    # Avg Distance
    top_countries_data = grouped_avg_distance[
        (grouped_avg_distance['relation'] == relation)
    ].head(10)
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_countries_data['country'], top_countries_data['avg_distance'])
    plt.xlabel('Country')
    plt.ylabel('Average Distance')
    plt.title(f'Top 10 Countries with Highest Avg Distance for Relation: {relation}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"paper_results_2/barplot_avg_distance_by_country_{relation}.png")
    plt.close()

    # Max Distance
    top_countries_data = grouped_max_distance[
        (grouped_max_distance['relation'] == relation)
    ].head(10)
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_countries_data['country'], top_countries_data['max_distance'])
    plt.xlabel('Country')
    plt.ylabel('Max Distance')
    plt.title(f'Top 10 Countries with Highest Max Distance for Relation: {relation}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"paper_results_2/barplot_max_distance_by_country_{relation}.png")
    plt.close()

    relation_data = new_df[new_df['relation'] == relation]
    
    plt.figure(figsize=(10, 6))
    sns.lmplot(
        x="hdi",
        y="citations",
        data=relation_data,
        scatter=False,
    )
    plt.xlabel('HDI')
    plt.ylabel('Citations')
    plt.title(f'Relationship between HDI and Citations for Relation: {relation}')
    plt.tight_layout()
    plt.savefig(f'paper_results_2/scatter_hdi_citations_{relation}.png')
    plt.close()

grouped_distances = new_df.groupby('relation')['avg_distance', 'max_distance', 'hdi'].mean()

# Create a text file to write the results
with open('paper_results_2/distance_results.txt', 'w') as file:
    file.write("Relation\tMean Avg Distance\tMean Max Distance\n")
    
    for relation, row in grouped_distances.iterrows():
        mean_avg_distance = row['avg_distance']
        mean_max_distance = row['max_distance']
        
        file.write(f"{relation}\t{mean_avg_distance:.2f}\t{mean_max_distance:.2f}\n")

# Create a text file to write the HDI results
with open('paper_results_2/hdi_results.txt', 'w') as file:
    file.write("Relation\tMean HDI\n")
    
    for relation, row in grouped_distances.iterrows():
        mean_hdi = row['hdi']
        
        file.write(f"{relation}\t{mean_hdi:.2f}\n")



collaborations = {
    "US": {},
    "EU": {},
    "CN": {},
    "US_EU": {},
    "US_CN": {},
    "CN_EU": {},
    "US_EU_CN": {},
    "OTHERS": {},
}

occurence_list = []

# Iterate over each row in the dataframe
for row in tqdm(df.itertuples()):
    country_list = set(literal_eval(row.countries))
    if len(set(country_list)) < 1:
        continue
    country_codes = []
    for country in country_list:
        country = "RS" if country == "XK" else country
        country = "CN" if country == "TW" else country
        if country in EU_COUNTRIES:
            country_codes.append("EU")
        else:
            country_codes.append(country)
    occurence_list.extend((country_code, row.year) for country_code in country_codes)

    if (
        "US" in country_codes
        and "EU" not in country_codes
        and "CN" not in country_codes
    ):
        for code in country_codes:
            if code != "US":
                collaborations["US"][code] = collaborations["US"].get(code, 0) + 1

    if (
        "EU" in country_codes
        and "US" not in country_codes
        and "CN" not in country_codes
    ):
        for code in country_codes:
            if code != "EU":
                collaborations["EU"][code] = collaborations["EU"].get(code, 0) + 1

    if (
        "CN" in country_codes
        and "US" not in country_codes
        and "EU" not in country_codes
    ):
        for code in country_codes:
            if code != "CN":
                collaborations["CN"][code] = collaborations["CN"].get(code, 0) + 1

    if "CN" in country_codes and "US" in country_codes and "EU" not in country_codes:
        for code in country_codes:
            if code != "CN" and code != "US":
                collaborations["US_CN"][code] = collaborations["US_CN"].get(code, 0) + 1

    if "CN" in country_codes and "US" not in country_codes and "EU" in country_codes:
        for code in country_codes:
            if code != "CN" and code != "EU":
                collaborations["CN_EU"][code] = collaborations["CN_EU"].get(code, 0) + 1

    if "CN" not in country_codes and "US" in country_codes and "EU" in country_codes:
        for code in country_codes:
            if code != "US" and code != "EU":
                collaborations["US_EU"][code] = collaborations["US_EU"].get(code, 0) + 1

    if "US" in country_codes and "EU" in country_codes and "CN" in country_codes:
        for code in country_codes:
            if code != "US" and code != "EU" and code != "CN":
                collaborations["US_EU_CN"][code] = (
                    collaborations["US_EU_CN"].get(code, 0) + 1
                )

    if (
        "US" not in country_codes
        and "EU" not in country_codes
        and "CN" not in country_codes
    ):
        for code in country_codes:
            collaborations["OTHERS"][code] = collaborations["OTHERS"].get(code, 0) + 1

# Count occurrences and create a DataFrame
all_countries = pd.DataFrame(occurence_list, columns=["country", "year"])
value_counts = (
    all_countries["country"]
    .value_counts()
    .rename_axis("country")
    .reset_index(name="count")
)

# Sort the DataFrame by count in descending order
value_counts = value_counts.sort_values("count", ascending=False)

# Plotting the bar plot for the top 15 values
top_values = value_counts.head(10)
top_values.plot(kind="bar", x="country", y="count", figsize=(10, 6))
plt.xlabel("Country")
plt.ylabel("Publications")
plt.title("Top 10 Values with the Most Publications")
plt.savefig(f"paper_results_2/bar_collaborations_by_country_top_10.png")
plt.close()

country_counts = (
    all_countries.groupby(["year", "country"]).size().reset_index(name="count")
)

# Get the top 15 most repeated countries for each year
top_countries_by_year = (
    country_counts.groupby("year")
    .apply(lambda x: x.nlargest(10, "count"))
    .reset_index(drop=True)
)

# Plotting line plot for the top 15 most repeated countries by year
fig, ax = plt.subplots(figsize=(10, 6))
for country, data in top_countries_by_year.groupby("country"):
    ax.plot(data["year"], data["count"], label=country)

ax.set_xlabel("Year")
ax.set_ylabel("Occurrences")
ax.set_title("Top 10 Most Repeated Countries by Year")
ax.legend()
plt.savefig(f"paper_results_2/lineplot_collaborations_by_country_per_year_top_10.png")
plt.close()

# Plotting parameters
num_top_countries = 10
bar_width = 0.5

# Plot each collaboration type
for collaboration, countries in collaborations.items():
    sorted_countries = sorted(countries.items(), key=lambda x: x[1], reverse=True)
    top_countries = sorted_countries[:num_top_countries]
    country_labels = [country[0] for country in top_countries]
    country_counts = [country[1] for country in top_countries]

    plt.figure(figsize=(10, 6))
    plt.bar(country_labels, country_counts, width=bar_width)
    plt.title(f"Top {num_top_countries} countries collaborating with {collaboration}")
    plt.xlabel("Country")
    plt.ylabel("Occurrences")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"paper_results_2/bar_num_collaborations_{collaboration}.png")
    plt.close()

    total_collaborations = sum(countries.values())
    country_percentages = [
        (count / total_collaborations) * 100 for count in country_counts
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(country_labels, country_percentages, width=bar_width)
    plt.title(f"Percentage of Collaborations for {collaboration} by Country")
    plt.xlabel("Country")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"paper_results_2/bar_percentage_collaborations_{collaboration}.png")
    plt.close()

    # Save the collaboration data to CSV
    with open(
        f"paper_results_2/num_collaborations_{collaboration}.csv", "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Country", "Occurrences"])
        writer.writerows(sorted_countries)

    data_with_percentage = [
        (country, count, count / total_collaborations * 100)
        for country, count in sorted_countries
    ]

    # Save the collaboration data to CSV
    with open(
        f"paper_results_2/percentage_collaborations_{collaboration}.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Country", "Percentage"])
        writer.writerows(data_with_percentage)

new_df = []
for row in tqdm(df.itertuples()):
    country_list = set(literal_eval(row.countries))
    if len(set(country_list)) < 1:
        continue
    
    # Replace codes and create an updated country list
    updated_list = [country_mapping.get(country, country) for country in country_list]
    if any(selected_country in updated_list for selected_country in selected_countries):
        for country in updated_list:
            new_df.append(
                    (
                        citations,
                        row.type,
                        row.year,
                        country,
                        row.max_distance,
                        row.avg_distance,
                        hdi
                    )
                )
            
new_df = pd.DataFrame(
    new_df,
    columns=[
        "citations",
        "type",
        "year",
        "country",
        "max_distance",
        "avg_distance",
        "hdi"
    ],
)

country_occurrences = new_df['country'].value_counts().head(10)

# Plot the top ten countries with the most occurrences
plt.figure(figsize=(10, 6))
country_occurrences.plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Number of Collaborations')
plt.title('Top Ten Countries with the Most Collaborations with CN, EU, US')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"paper_results_2/bar_plot_most_collaborations_with_cn_eu_us.png")
plt.close()
