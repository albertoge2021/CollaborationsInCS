from ast import literal_eval
from collections import Counter, defaultdict
import json
from matplotlib import ticker
import networkx as nx
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import squarify
import pandas as pd
import pycountry_convert as pc
import scipy.stats as stats
import seaborn as sns
import warnings


warnings.simplefilter(action="ignore", category=FutureWarning)

#df = pd.read_csv("data_countries/cs_works.csv")

selected_countries = ["US", "EU", "CN"]
colors= {
    "Mixed": "deepskyblue",
    "US-EU-CN": "limegreen",
    "Others": "orangered",
}
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

CN_COUNTRIES = [
    "CN",
    "HK",
    "MO",
    "TW",
]
"""
new_df_list = []
countries_df_list = []
concepts_df_list = []

for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    country_list = []
    countries = literal_eval(row.countries)
    if len(set(countries)) < 2:
        continue
    else:
        countries = set(countries)

    # Check if all values are in ["US", "EU", "CN"]
    if all(country in ["US", "EU", "CN"] for country in country_list):
        relation = "US-EU-CN"
    elif any(country in ["US", "EU", "CN"] for country in country_list):
        relation = "Mixed"
    else:
        relation = "Others"
    for country in countries:
        if country in EU_COUNTRIES:
            country = "EU"
        if country in CN_COUNTRIES:
            country = "CN"
        country_list.append(country)
        countries_df_list.append(
            (
                row.citations,
                row.type,
                row.publication_year,
                country,
            )
        )

    for concept in literal_eval(row.concepts):
        concepts_df_list.append(
            (
                concept,
                row.publication_year,
                row.citations,
                row.type,
                row.countries,
                relation
            )
        )

    new_df_list.append(
        (
            #row.title,
            row.publication_year,
            row.citations,
            row.type,
            row.is_retracted,
            row.institution_types,
            row.countries,
            row.concepts,
            row.gdp,
            row.hdi,
            row.hemisphere,
            relation,
        )
    )

country_df = pd.DataFrame(
    countries_df_list,
    columns=[
        "citations",
        "type",
        "publication_year",
        "country",
    ],
)

df_concepts = pd.DataFrame(
    concepts_df_list,
    columns=[
        "concept",
        "publication_year",
        "citations",
        "type",
        "countries",
        "relation",
    ],
)

# Create a DataFrame using pib_data and columns
new_df = pd.DataFrame(
    new_df_list,
    columns=[
        #"title",
        "publication_year",
        "citations",
        "type",
        "is_retracted",
        "institution_types",
        "countries",
        "concepts",
        "gdp",
        "hdi",
        "hemisphere",
        "relation",
    ],
)

country_df.to_csv("country_df.csv", index=False)
df_concepts.to_csv("df_concepts.csv", index=False)
new_df.to_csv("new_df.csv", index=False)"""

# Retrieve DataFrames from CSV files
#country_df = pd.read_csv("country_df.csv")
#df_concepts = pd.read_csv("df_concepts.csv")
new_df = pd.read_csv("new_df.csv")

""" TODO

US-EU-CN

Collaborations between these countries and the others. Number and citations. Provide data with only the others to compare. DONE

		- Measure the difference by HDI. Does it impact redaction rate? Provide also US-EU-CN rate and Only others rate. DONE
		- Measure the difference by GDP. Does it impact redaction rate? Provide also US-EU-CN rate and Only others rate. DISCARDED
		- Publication trends in publication type?
		- Most prolific instutution types, most relevant.
		- Topics of the groups?
		- Median and average percentage of participants. """


# Line plot for number of collaborations per year
counts = new_df.groupby(["relation", "publication_year"]).size().reset_index(name="count")
sns.lineplot(data=counts, x="publication_year", y="count", hue="relation", palette=colors)
plt.xlabel("Year")
plt.ylabel("Number of collaborations")
plt.title("Number of collaborations by regions per year")
plt.savefig(f"results/lineplot_collaborations_per_year_per_collaboration.png")
plt.close()

new_df['citations'] = pd.to_numeric(new_df['citations'], errors='coerce')

# Bar plot for mean citations
plt.figure(figsize=(10, 6))
sns.barplot(x='relation', y='citations', data=new_df, estimator='mean', ci=None, palette=colors)
plt.title('Mean Citations by Relation')
plt.xlabel('Relation')
plt.ylabel('Mean Citations')
plt.savefig(f"results/barplot_mean_citations_per_relation.png")
plt.close()

# Bar plot for median citations
plt.figure(figsize=(10, 6))
sns.barplot(x='relation', y='citations', data=new_df, estimator='median', ci=None, palette=colors)
plt.title('Median Citations by Relation')
plt.xlabel('Relation')
plt.ylabel('Median Citations')
plt.savefig(f"results/barplot_median_citations_per_relation.png")
plt.close()

# Classify 'hdi' values into bins
bins = [0, 0.549, 0.699, 0.799, 1.0]
labels = ["Low", "Medium", "High", "Very High"]
new_df['hdi_class'] = pd.cut(new_df['mean_hdi'], bins=bins, labels=labels, right=False)

# Calculate average citations for each 'hdi_class' and 'relation'
avg_citations = new_df.groupby(['hdi_class', 'relation'])['citations'].mean().reset_index(name='avg_citations')

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='hdi_class', y='avg_citations', hue='relation', data=avg_citations, palette=colors)
plt.xlabel('HDI Classification')
plt.ylabel('Average Citations')
plt.title('Average Citations by HDI Classification and Relation')
plt.legend(title='Relation', loc='upper right')
plt.savefig(f"results/barplot_hdi_mean_citations_per_relation.png")
plt.close()

result = new_df.groupby(['hdi_class', 'relation']).size().reset_index(name='count')

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='hdi_class', y='count', hue='relation', data=result, palette=colors)
plt.xlabel('HDI Classification')
plt.ylabel('Number of Collaborations')
plt.title('Number of Collaborations by HDI Classification and Relation')
plt.legend(title='Relation', loc='upper right')
plt.savefig(f"results/barplot_hdi_collaborations_per_relation.png")
plt.close()

result = new_df.groupby(['hdi_class', 'relation']).size().reset_index(name='count')

# Calculate the total count for each 'hdi_class'
total_counts = result.groupby('hdi_class')['count'].sum()

# Calculate the percentage within each 'hdi_class'
result['percentage'] = result.apply(lambda row: row['count'] / total_counts[row['hdi_class']] * 100, axis=1)

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='hdi_class', y='percentage', hue='relation', data=result, palette=colors)
plt.xlabel('HDI Classification')
plt.ylabel('Percentage of Collaborations')
plt.title('Percentage of Collaborations by HDI Classification and Relation')
plt.legend(title='Relation', loc='upper right')
plt.savefig(f"results/barplot_hdi_percentage_collaborations_per_relation.png")
plt.close()

# Step 1: Calculate the number of retracted articles, total number of rows, and percentage by relation
retraction_stats = new_df.groupby('relation')['is_retracted'].agg(['sum', 'count']).reset_index()
retraction_stats['percentage'] = (retraction_stats['sum'] / retraction_stats['count']) * 100

# Step 2: Save the results to a CSV file
retraction_stats.to_csv('results/retraction_stats.csv', index=False)

# Step 3: Plot the percentages in a bar plot by relation
plt.figure(figsize=(10, 6))
sns.barplot(x='relation', y='percentage', data=retraction_stats, palette=colors)
plt.xlabel('Relation')
plt.ylabel('Percentage of Retracted Articles')
plt.title('Percentage of Retracted Articles by Relation')
plt.savefig(f"results/barplot_retracted_publications_per_relation.png")
plt.close()

retraction_stats = new_df.groupby(['relation', 'publication_year'])['is_retracted'].agg(['sum', 'count']).reset_index()
retraction_stats['percentage'] = (retraction_stats['sum'] / retraction_stats['count']) * 100

# Step 2: Plot the percentages in a bar plot by relation and year
plt.figure(figsize=(12, 6))
sns.lineplot(x='publication_year', y='percentage', hue='relation', data=retraction_stats, palette=colors)
plt.xlabel('Publication Year')
plt.ylabel('Percentage of Retracted Articles')
plt.title('Percentage of Retracted Articles by Relation and Year')
plt.legend(title='Relation', loc='upper right')
plt.savefig(f"results/lineplot_retracted_publications_per_relation_by_year.png")
plt.close()

# Step 1: Find the top 5 most common types by relation
top_types_by_relation = new_df.groupby(['relation', 'type']).size().reset_index(name='count')
top_types_by_relation = top_types_by_relation.sort_values(by=['relation', 'count'], ascending=[True, False])
top_types_by_relation = top_types_by_relation.groupby('relation').head(5)

# Step 2: Plot the top 5 most common types by relation
plt.figure(figsize=(12, 6))
sns.barplot(x='type', y='count', hue='relation', data=top_types_by_relation, palette=colors)
plt.xlabel('Publication Type')
plt.ylabel('Count')
plt.title('Top 5 Most Common Types by Relation')
plt.legend(title='Relation', loc='upper right')
plt.savefig(f"results/barplot_type_per_relation.png")
plt.close()

top_types_by_relation = new_df.groupby(['relation', 'type']).size().reset_index(name='count')
top_types_by_relation = top_types_by_relation.sort_values(by=['relation', 'count'], ascending=[True, False])
top_types_by_relation = top_types_by_relation.groupby('relation').head(5)

# Calculate the total count for each 'relation'
total_counts = top_types_by_relation.groupby('relation')['count'].sum()

# Calculate the percentage within each 'relation'
top_types_by_relation['percentage'] = top_types_by_relation.apply(lambda row: row['count'] / total_counts[row['relation']] * 100, axis=1)

# Step 2: Plot the percentage for the top 5 most common types by relation
plt.figure(figsize=(12, 6))
sns.barplot(x='type', y='percentage', hue='relation', data=top_types_by_relation, palette=colors)
plt.xlabel('Publication Type')
plt.ylabel('Percentage')
plt.title('Top 5 Most Common Types by Relation (Percentage)')
plt.legend(title='Relation', loc='upper right')
plt.savefig("results/barplot_type_percentage_per_relation.png")
plt.close()

institution_types_count = {}
concept_count = {}
concept_retraction_count = {}
concept_count_by_relation_year = {}

# Iterate over rows and count institution types and concepts for each relation
for row in tqdm(new_df.itertuples(), total=len(new_df), desc="Counting Institution Types and Concepts"):
    relation = row.relation
    year = row.publication_year
    
    # Count institution types
    institution_types = set(literal_eval(row.institution_types)) if pd.notnull(row.institution_types) else []
    for institution_type in institution_types:
        key = (relation, year, institution_type)
        institution_types_count[key] = institution_types_count.get(key, 0) + 1
    
    # Count concepts
    concepts = set(literal_eval(row.concepts)) if pd.notnull(row.concepts) else []
    for concept in concepts:
        key = (relation, year, concept)
        concept_count[key] = concept_count.get(key, 0) + 1

        is_retracted = row.is_retracted
        concept_retraction_key = (relation, year, concept)
        concept_retraction_count.setdefault(concept_retraction_key, {'total': 0, 'retracted': 0})
        concept_retraction_count[concept_retraction_key]['total'] += 1
        concept_retraction_count[concept_retraction_key]['retracted'] += int(is_retracted)

        concept_count_by_relation_year[key] = concept_count_by_relation_year.get(key, 0) + 1

# Convert dictionaries to DataFrames
top_institution_types_by_relation = pd.DataFrame(list(institution_types_count.items()), columns=['relation_year', 'institution_type_count'])
top_concepts_by_relation = pd.DataFrame(list(concept_count.items()), columns=['relation_year', 'concept_count'])

# Extract relation, year, and institution_type/concept from the 'relation_year' column
top_institution_types_by_relation[['relation', 'year', 'institution_type']] = pd.DataFrame(top_institution_types_by_relation['relation_year'].tolist(), index=top_institution_types_by_relation.index)
top_concepts_by_relation[['relation', 'year', 'concept']] = pd.DataFrame(top_concepts_by_relation['relation_year'].tolist(), index=top_concepts_by_relation.index)

# Save the DataFrames to CSV files
top_institution_types_by_relation.to_csv('results/top_institution_types_by_relation.csv', index=False)
top_concepts_by_relation.to_csv('results/top_concepts_by_relation.csv', index=False)

# Find the top 5 most common institution types and concepts by relation
top_institution_types_by_relation = top_institution_types_by_relation.groupby('relation').apply(lambda group: group.nlargest(5, 'institution_type_count')).reset_index(drop=True)
top_concepts_by_relation = top_concepts_by_relation.groupby('relation').apply(lambda group: group.nlargest(5, 'concept_count')).reset_index(drop=True)

# Convert concept_retraction_count dictionary to a DataFrame
concept_retraction_df = pd.DataFrame(list(concept_retraction_count.items()), columns=['relation_concept', 'counts'])

# Extract relation and concept from the 'relation_concept' column
concept_retraction_df[['relation', 'year', 'concept']] = pd.DataFrame(concept_retraction_df['relation_concept'].tolist(), index=concept_retraction_df.index)
excluded_concepts = ['Computer science']  # Add the computer science concepts you want to exclude
concept_retraction_df = concept_retraction_df[~concept_retraction_df['concept'].isin(excluded_concepts)]

# Calculate the percentage of retractions for each concept
concept_retraction_df['percentage'] = concept_retraction_df['counts'].apply(lambda x: x['retracted'] / x['total'] * 100)

# Filter concepts with more than 10 retractions
concept_retraction_df = concept_retraction_df[concept_retraction_df['counts'].apply(lambda x: x['retracted'] > 10)]

# Sort the DataFrame by the percentage of retractions in descending order
concept_retraction_df = concept_retraction_df.sort_values(by='percentage', ascending=False)

# Save the results to a CSV file
concept_retraction_df.to_csv('results/concept_retraction_counts.csv', index=False)

# Convert the concept_count_by_relation_year dictionary to a DataFrame
concept_count_df = pd.DataFrame(list(concept_count_by_relation_year.items()), columns=['relation_year_concept', 'count'])

# Extract relation, year, and concept from the 'relation_year_concept' column
concept_count_df[['relation', 'year', 'concept']] = pd.DataFrame(concept_count_df['relation_year_concept'].tolist(), index=concept_count_df.index)
concept_count_df = concept_count_df[~concept_count_df['concept'].isin(excluded_concepts)]

# Find the top 10 most frequent concepts for each relation and year
top_concepts_by_relation_year = concept_count_df.groupby(['relation', 'year']).apply(lambda group: group.nlargest(10, 'count')).reset_index(drop=True)

concept_colors = {
    'Physics': 'red',
    'Artificial intelligence': 'blue',
    'Programming language': 'green',
    'Algorithm': 'orange',
    'Telecommunications': 'purple',
    'Materials science': 'cyan',
    'Psychology': 'magenta',
    'Biology': 'brown',
    'Mathematics': 'pink',
    'Engineering': 'gray',
    'Quantum mechanics': 'lime',
    'Chemistry': 'olive',
    'Operating system': 'maroon',
    'Medicine': 'navy',
    'Economics': 'teal',
    'Mathematical analysis': 'teal',
    'Optics': 'teal',
}

# Plot the top 10 most frequent concepts for each relation by year
for relation in new_df['relation'].unique():
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='year', y='count', hue='concept', data=top_concepts_by_relation_year[(top_concepts_by_relation_year['relation'] == relation) & (top_concepts_by_relation_year['count'] > 0)])
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title(f'Top 10 Most Frequent Concepts for Relation: {relation}')
    plt.legend(title='Concept', loc='upper right')
    plt.savefig(f"results/lineplot_top_concepts_{relation}.png")
    plt.close()

# Plot the top 5 most common institution types by relation
plt.figure(figsize=(12, 6))
sns.barplot(x='institution_type', y='institution_type_count', hue='relation', data=top_institution_types_by_relation)
plt.xlabel('Institution Type')
plt.ylabel('Count')
plt.title('Top 5 Most Common Institution Types by Relation')
plt.legend(title='Relation', loc='upper right')
plt.savefig("results/barplot_institution_type_per_relation.png")
plt.close()

mixed_df = new_df[new_df['relation'] == 'Mixed']

country_counts = {}
country_citations = {}
for row in tqdm(mixed_df.itertuples(), total=len(mixed_df), desc="Counting Countries for Mixed Relation"):
    countries = literal_eval(row.countries)
    for country in countries:
        if country in EU_COUNTRIES:
            country = "EU"
        elif country in CN_COUNTRIES:
            country = "CN"
        country_counts[country] = country_counts.get(country, 0) + 1
        country_citations[country] = country_citations.get(country, 0) + row.citations

# Create a DataFrame with country counts and citations
country_stats_df = pd.DataFrame({
    'Country': list(country_counts.keys()),
    'Count': list(country_counts.values()),
    'Total Citations': list(country_citations.values()),
})

# Calculate average citations
country_stats_df['Average Citations'] = country_stats_df['Total Citations'] / country_stats_df['Count']

country_stats_df = country_stats_df.sort_values(by='Average Citations', ascending=False)

# Save the whole list to CSV
country_stats_df.to_csv('results/all_countries_mixed_relation.csv', index=False)

# Select top 10 countries by count
top_countries_by_count = country_stats_df.nlargest(10, 'Count')

# Plot barplot for top 10 countries by count
plt.figure(figsize=(10, 6))
plt.bar(top_countries_by_count['Country'], top_countries_by_count['Count'])
plt.xlabel('Country')
plt.ylabel('Publications')
plt.title('Top 10 Most Published Countries for Mixed Relation')
plt.xticks(rotation=45, ha='right')
plt.savefig("results/bar_plot_most_published_countries.png")
plt.close()

# Select top 10 countries by average citation
top_countries_by_citations = country_stats_df.nlargest(10, 'Average Citations')

# Plot barplot for top 10 countries by average citation
plt.figure(figsize=(10, 6))
plt.bar(top_countries_by_citations['Country'], top_countries_by_citations['Average Citations'])
plt.xlabel('Country')
plt.ylabel('Average Citations')
plt.title('Top 10 Countries by Average Citation for Mixed Relation')
plt.xticks(rotation=45, ha='right')
plt.savefig("results/bar_plot_highest_avg_citations.png")
plt.close()
