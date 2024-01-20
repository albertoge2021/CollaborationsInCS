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
from scipy.stats import shapiro


warnings.simplefilter(action="ignore", category=FutureWarning)

# df = pd.read_csv("data_countries/cs_works.csv")

selected_countries = ["US", "EU", "CN"]
colors = {
    "Mixed": "deepskyblue",
    "US-EU-CN": "limegreen",
    "Other Countries": "orangered",
}
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
    "XKG",
    "TWN",
]
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
# country_df = pd.read_csv("country_df.csv")
# df_concepts = pd.read_csv("df_concepts.csv")
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

result_by_collaborators = {
    "collaborators": [],
    "retracted_count": [],
    "non_retracted_count": [],
    "percentage_retracted": [],
}
total_by_type = defaultdict(int)

# Iterate through the DataFrame rows with tqdm
for row in tqdm(new_df.itertuples(), total=len(new_df), desc="Counting Collaborators"):
    countries = literal_eval(row.countries)
    is_retracted = row.is_retracted

    # Count the number of collaborators
    num_collaborators = len(countries)

    # Update result_by_collaborators dictionary
    if num_collaborators not in result_by_collaborators["collaborators"]:
        result_by_collaborators["collaborators"].append(num_collaborators)
        result_by_collaborators["retracted_count"].append(0)
        result_by_collaborators["non_retracted_count"].append(0)

    index = result_by_collaborators["collaborators"].index(num_collaborators)
    if is_retracted:
        result_by_collaborators["retracted_count"][index] += 1
    else:
        result_by_collaborators["non_retracted_count"][index] += 1

# Filter data for collaborators up to 20
filter_limit = 20
filtered_data = {
    "collaborators": [],
    "percentage_retracted": [],
}

for i, collaborators in enumerate(result_by_collaborators["collaborators"]):
    filtered_data["collaborators"].append(collaborators)
    retracted_count = result_by_collaborators["retracted_count"][i]
    non_retracted_count = result_by_collaborators["non_retracted_count"][i]
    percentage_retracted = (
        (retracted_count / (retracted_count + non_retracted_count)) * 100
        if (retracted_count + non_retracted_count) > 0
        else 0
    )
    filtered_data["percentage_retracted"].append(percentage_retracted)


# Convert data to a DataFrame
df_filtered = pd.DataFrame(filtered_data)

# Check for normality using Shapiro-Wilk test
stat, shapiro_p_value = shapiro(df_filtered['percentage_retracted'])

# Perform Pearson or Spearman correlation test based on normality result
if shapiro_p_value > 0.05:  # If p-value > 0.05, assume normal distribution
    correlation_method = "Pearson"
    correlation_test = pearsonr
else:
    correlation_method = "Spearman"
    correlation_test = spearmanr

# Perform correlation test
corr, p_value = correlation_test(df_filtered['collaborators'], df_filtered['percentage_retracted'])

# Save results in APA format to a text file
with open("results/correlation_results_for_collaborators.txt", "w") as f:
    f.write(
        f"{correlation_method} Correlation: r({len(df_filtered) - 2}) = {corr:.3f}, p = {p_value:.3f}\n"
    )

df_filtered = df_filtered[df_filtered["collaborators"] <= filter_limit]

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x="collaborators", y="percentage_retracted")
plt.title("Percentage of Retracted Works by Number of Collaborators (up to 20)")
plt.xlabel("Number of Collaborators")
plt.ylabel("Percentage of Retraction")
plt.xticks(range(1, filter_limit + 1))  # Set x-axis ticks to integer values
plt.grid(True)
plt.savefig(f"results/lineplot_correlation_collaborators_retraction.png")
plt.close()

for relation in new_df["relation"].unique():
    # Filter the DataFrame by 'relation'
    relation_df = new_df[new_df["relation"] == relation]
    result_by_collaborators = {
        "collaborators": [],
        "retracted_count": [],
        "non_retracted_count": [],
        "percentage_retracted": [],
    }
    total_by_type = defaultdict(int)

    # Iterate through the DataFrame rows with tqdm
    for row in tqdm(
        relation_df.itertuples(), total=len(relation_df), desc="Counting Collaborators"
    ):
        countries = literal_eval(row.countries)
        is_retracted = row.is_retracted

        # Count the number of collaborators
        num_collaborators = len(countries)

        # Update result_by_collaborators dictionary
        if num_collaborators not in result_by_collaborators["collaborators"]:
            result_by_collaborators["collaborators"].append(num_collaborators)
            result_by_collaborators["retracted_count"].append(0)
            result_by_collaborators["non_retracted_count"].append(0)

        index = result_by_collaborators["collaborators"].index(num_collaborators)
        if is_retracted:
            result_by_collaborators["retracted_count"][index] += 1
        else:
            result_by_collaborators["non_retracted_count"][index] += 1

    # Filter data for collaborators up to 20
    filter_limit = 20
    filtered_data = {
        "collaborators": [],
        "percentage_retracted": [],
    }

    for i, collaborators in enumerate(result_by_collaborators["collaborators"]):
        filtered_data["collaborators"].append(collaborators)
        retracted_count = result_by_collaborators["retracted_count"][i]
        non_retracted_count = result_by_collaborators["non_retracted_count"][i]
        percentage_retracted = (
            (retracted_count / (retracted_count + non_retracted_count)) * 100
            if (retracted_count + non_retracted_count) > 0
            else 0
        )
        filtered_data["percentage_retracted"].append(percentage_retracted)

    # Convert data to a DataFrame
    df_filtered = pd.DataFrame(filtered_data)

    # Check for normality using Shapiro-Wilk test
    stat, shapiro_p_value = shapiro(df_filtered['percentage_retracted'])

    # Perform Pearson or Spearman correlation test based on normality result
    if shapiro_p_value > 0.05:  # If p-value > 0.05, assume normal distribution
        correlation_method = "Pearson"
        correlation_test = pearsonr
    else:
        correlation_method = "Spearman"
        correlation_test = spearmanr

    # Perform correlation test
    corr, p_value = correlation_test(df_filtered['collaborators'], df_filtered['percentage_retracted'])

    # Save results in APA format to a text file
    with open(f"results/correlation_results_for_collaborators_{relation}.txt", "w") as f:
        f.write(
            f"{correlation_method} Correlation: r({len(df_filtered) - 2}) = {corr:.3f}, p = {p_value:.3f}\n"
        )

    df_filtered = df_filtered[df_filtered["collaborators"] <= filter_limit]

    # Plotting
    if relation == "US-EU-CN":
        relation_name = "US, EU, and CN"
    elif relation == "Mixed":
        relation_name = "Mixed"
    else:
        relation_name = "Other Countries"
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x="collaborators", y="percentage_retracted")
    plt.title(
        f"Percentage of Retracted Works by Number of Collaborators (up to 20) for {relation_name}"
    )
    plt.xlabel("Number of Collaborators")
    plt.ylabel("Percentage of Retraction")
    plt.xticks(range(1, filter_limit + 1))  # Set x-axis ticks to integer values
    plt.grid(True)
    plt.savefig(f"results/lineplot_correlation_collaborators_retraction_{relation}.png")
    plt.close()

# Group by 'relation'
grouped_by_relation = new_df.groupby("relation")

# Define your groups for t-tests
groups = new_df["relation"].unique()

# Create a file to save the t-test results
output_file = "results/t_test_results.txt"

# Open the file in write mode
with open(output_file, "w") as file:
    # Perform t-tests for each pair of groups
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = groups[i]
            group2 = groups[j]

            # Extract data for the two groups
            data_group1 = new_df[new_df["relation"] == group1]["citations"]
            data_group2 = new_df[new_df["relation"] == group2]["citations"]

            # Perform t-test
            t_statistic, p_value = ttest_ind(data_group1, data_group2)

            # Write results to the file
            file.write(f"T-test between {group1} and {group2}:\n")
            file.write(f"T-statistic: {t_statistic}\n")
            file.write(f"P-value: {p_value}\n")
            file.write("\n")

# Save a txt file with the number of rows, number of is_retracted=True,
# and the percentage of is_retracted=True / total rows for each group
# Perform a statistical test to check for differences in average citations between groups
grouped_citations = [group["citations"] for name, group in grouped_by_relation]

# Perform ANOVA
anova_result = stats.f_oneway(*grouped_citations)

# Extract relevant information
degrees_of_freedom_between = len(grouped_citations) - 1
degrees_of_freedom_within = len(grouped_citations[0]) - degrees_of_freedom_between - 1
f_statistic = anova_result.statistic
p_value = anova_result.pvalue

# Print the formatted result
result_string = f"F({degrees_of_freedom_between}, {degrees_of_freedom_within}) = {f_statistic:.2f}, p = {p_value:.3f}"

with open("results/grouped_stats.txt", "w") as f:
    # Print the ANOVA result
    f.write("ANOVA Result for citations:")
    f.write(str(result_string) + "\n")
    for name, group in grouped_by_relation:
        num_rows = len(group)
        num_retracted = group["is_retracted"].sum()
        percentage_retracted = (num_retracted / num_rows) * 100 if num_rows > 0 else 0

        f.write(f"Group: {name}\n")
        f.write(f"Number of Rows: {num_rows}\n")
        f.write(f"Number of is_retracted=True: {num_retracted}\n")
        f.write(f"Percentage of is_retracted=True: {percentage_retracted:.2f}%\n\n")

# Calculate average and median citations for each group
grouped_stats = grouped_by_relation["citations"].agg(["mean", "median"])

# Save the grouped statistics to a CSV file
grouped_stats.to_csv("results/grouped_citations_stats.csv")

# Line plot for number of collaborations per year
counts = (
    new_df.groupby(["relation", "publication_year"]).size().reset_index(name="count")
)
sns.lineplot(
    data=counts, x="publication_year", y="count", hue="relation", palette=colors
)
plt.xlabel("Year")
plt.ylabel("Number of collaborations")
plt.title("Number of collaborations by regions per year")
plt.savefig(f"results/lineplot_collaborations_per_year_per_collaboration.png")
plt.close()

new_df["citations"] = pd.to_numeric(new_df["citations"], errors="coerce")

# Bar plot for mean citations
plt.figure(figsize=(10, 6))
sns.barplot(
    x="relation", y="citations", data=new_df, estimator="mean", ci=None, palette=colors
)
plt.title("Mean Citations by Relation")
plt.xlabel("Relation")
plt.ylabel("Mean Citations")
plt.savefig(f"results/barplot_mean_citations_per_relation.png")
plt.close()

# Bar plot for median citations
plt.figure(figsize=(10, 6))
sns.barplot(
    x="relation",
    y="citations",
    data=new_df,
    estimator="median",
    ci=None,
    palette=colors,
)
plt.title("Median Citations by Relation")
plt.xlabel("Relation")
plt.ylabel("Median Citations")
plt.savefig(f"results/barplot_median_citations_per_relation.png")
plt.close()

# Classify 'hdi' values into bins
bins = [0, 0.549, 0.699, 0.799, 1.0]
labels = ["Low", "Medium", "High", "Very High"]
new_df["hdi_class"] = pd.cut(new_df["mean_hdi"], bins=bins, labels=labels, right=False)

# Calculate average citations for each 'hdi_class' and 'relation'
avg_citations = (
    new_df.groupby(["hdi_class", "relation"])["citations"]
    .mean()
    .reset_index(name="avg_citations")
)

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(
    x="hdi_class", y="avg_citations", hue="relation", data=avg_citations, palette=colors
)
plt.xlabel("HDI Classification")
plt.ylabel("Average Citations")
plt.title("Average Citations by HDI Classification and Relation")
plt.legend(title="Relation", loc="upper right")
plt.savefig(f"results/barplot_hdi_mean_citations_per_relation.png")
plt.close()

result = new_df.groupby(["hdi_class", "relation"]).size().reset_index(name="count")

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x="hdi_class", y="count", hue="relation", data=result, palette=colors)
plt.xlabel("HDI Classification")
plt.ylabel("Number of Collaborations")
plt.title("Number of Collaborations by HDI Classification and Relation")
plt.legend(title="Relation", loc="upper right")
plt.savefig(f"results/barplot_hdi_collaborations_per_relation.png")
plt.close()

result = new_df.groupby(["hdi_class", "relation"]).size().reset_index(name="count")

# Calculate the total count for each 'hdi_class'
total_counts = result.groupby("hdi_class")["count"].sum()

# Calculate the percentage within each 'hdi_class'
result["percentage"] = result.apply(
    lambda row: row["count"] / total_counts[row["hdi_class"]] * 100, axis=1
)

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x="hdi_class", y="percentage", hue="relation", data=result, palette=colors)
plt.xlabel("HDI Classification")
plt.ylabel("Percentage of Collaborations")
plt.title("Percentage of Collaborations by HDI Classification and Relation")
plt.legend(title="Relation", loc="upper right")
plt.savefig(f"results/barplot_hdi_percentage_collaborations_per_relation.png")
plt.close()

# Step 1: Calculate the number of retracted articles, total number of rows, and percentage by relation
retraction_stats = (
    new_df.groupby("relation")["is_retracted"].agg(["sum", "count"]).reset_index()
)
retraction_stats["percentage"] = (
    retraction_stats["sum"] / retraction_stats["count"]
) * 100

# Step 2: Save the results to a CSV file
retraction_stats.to_csv("results/retraction_stats.csv", index=False)

# Step 3: Plot the percentages in a bar plot by relation
plt.figure(figsize=(10, 6))
sns.barplot(x="relation", y="percentage", data=retraction_stats, palette=colors)
plt.xlabel("Relation")
plt.ylabel("Percentage of Retracted Articles")
plt.title("Percentage of Retracted Articles by Relation")
plt.savefig(f"results/barplot_retracted_publications_per_relation.png")
plt.close()

retraction_stats = (
    new_df.groupby(["relation", "publication_year"])["is_retracted"]
    .agg(["sum", "count"])
    .reset_index()
)
retraction_stats["percentage"] = (
    retraction_stats["sum"] / retraction_stats["count"]
) * 100

# Step 2: Plot the percentages in a bar plot by relation and year
plt.figure(figsize=(12, 6))
sns.lineplot(
    x="publication_year",
    y="percentage",
    hue="relation",
    data=retraction_stats,
    palette=colors,
)
plt.xlabel("Publication Year")
plt.ylabel("Percentage of Retracted Articles")
plt.title("Percentage of Retracted Articles by Relation and Year")
plt.legend(title="Relation", loc="upper right")
plt.savefig(f"results/lineplot_retracted_publications_per_relation_by_year.png")
plt.close()

# Step 1: Find the top 5 most common types by relation
top_types_by_relation = (
    new_df.groupby(["relation", "type"]).size().reset_index(name="count")
)
top_types_by_relation = top_types_by_relation.sort_values(
    by=["relation", "count"], ascending=[True, False]
)
top_types_by_relation = top_types_by_relation.groupby("relation").head(5)

# Step 2: Plot the top 5 most common types by relation
plt.figure(figsize=(12, 6))
sns.barplot(
    x="type", y="count", hue="relation", data=top_types_by_relation, palette=colors
)
plt.xlabel("Publication Type")
plt.ylabel("Count")
plt.title("Top 5 Most Common Types by Relation")
plt.legend(title="Relation", loc="upper right")
plt.savefig(f"results/barplot_type_per_relation.png")
plt.close()

top_types_by_relation = (
    new_df.groupby(["relation", "type"]).size().reset_index(name="count")
)
top_types_by_relation = top_types_by_relation.sort_values(
    by=["relation", "count"], ascending=[True, False]
)
top_types_by_relation = top_types_by_relation.groupby("relation").head(5)

# Calculate the total count for each 'relation'
total_counts = top_types_by_relation.groupby("relation")["count"].sum()

# Calculate the percentage within each 'relation'
top_types_by_relation["percentage"] = top_types_by_relation.apply(
    lambda row: row["count"] / total_counts[row["relation"]] * 100, axis=1
)

# Step 2: Plot the percentage for the top 5 most common types by relation
plt.figure(figsize=(12, 6))
sns.barplot(
    x="type", y="percentage", hue="relation", data=top_types_by_relation, palette=colors
)
plt.xlabel("Publication Type")
plt.ylabel("Percentage")
plt.title("Top 5 Most Common Types by Relation (Percentage)")
plt.legend(title="Relation", loc="upper right")
plt.savefig("results/barplot_type_percentage_per_relation.png")
plt.close()

"""institution_types_count = {}
concept_count = {}
concept_retraction_count = {}
concept_count_by_relation_year = {}

# Iterate over rows and count institution types and concepts for each relation
for row in tqdm(
    new_df.itertuples(),
    total=len(new_df),
    desc="Counting Institution Types and Concepts",
):
    relation = row.relation
    year = row.publication_year

    # Count institution types
    institution_types = (
        set(literal_eval(row.institution_types))
        if pd.notnull(row.institution_types)
        else []
    )
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
        concept_retraction_count.setdefault(
            concept_retraction_key, {"total": 0, "retracted": 0}
        )
        concept_retraction_count[concept_retraction_key]["total"] += 1
        concept_retraction_count[concept_retraction_key]["retracted"] += int(
            is_retracted
        )

        concept_count_by_relation_year[key] = (
            concept_count_by_relation_year.get(key, 0) + 1
        )

# Convert dictionaries to DataFrames
top_institution_types_by_relation = pd.DataFrame(
    list(institution_types_count.items()),
    columns=["relation_year", "institution_type_count"],
)
top_concepts_by_relation = pd.DataFrame(
    list(concept_count.items()), columns=["relation_year", "concept_count"]
)

# Extract relation, year, and institution_type/concept from the 'relation_year' column
top_institution_types_by_relation[
    ["relation", "year", "institution_type"]
] = pd.DataFrame(
    top_institution_types_by_relation["relation_year"].tolist(),
    index=top_institution_types_by_relation.index,
)
top_concepts_by_relation[["relation", "year", "concept"]] = pd.DataFrame(
    top_concepts_by_relation["relation_year"].tolist(),
    index=top_concepts_by_relation.index,
)

# Save the DataFrames to CSV files
top_institution_types_by_relation.to_csv(
    "results/top_institution_types_by_relation.csv", index=False
)
top_concepts_by_relation.to_csv("results/top_concepts_by_relation.csv", index=False)

# Find the top 5 most common institution types and concepts by relation
top_institution_types_by_relation = (
    top_institution_types_by_relation.groupby("relation")
    .apply(lambda group: group.nlargest(5, "institution_type_count"))
    .reset_index(drop=True)
)
top_concepts_by_relation = (
    top_concepts_by_relation.groupby("relation")
    .apply(lambda group: group.nlargest(5, "concept_count"))
    .reset_index(drop=True)
)

# Convert concept_retraction_count dictionary to a DataFrame
concept_retraction_df = pd.DataFrame(
    list(concept_retraction_count.items()), columns=["relation_concept", "counts"]
)

# Extract relation and concept from the 'relation_concept' column
concept_retraction_df[["relation", "year", "concept"]] = pd.DataFrame(
    concept_retraction_df["relation_concept"].tolist(),
    index=concept_retraction_df.index,
)
excluded_concepts = [
    "Computer science"
]  # Add the computer science concepts you want to exclude
concept_retraction_df = concept_retraction_df[
    ~concept_retraction_df["concept"].isin(excluded_concepts)
]

# Calculate the percentage of retractions for each concept
concept_retraction_df["percentage"] = concept_retraction_df["counts"].apply(
    lambda x: x["retracted"] / x["total"] * 100
)

# Filter concepts with more than 10 retractions
concept_retraction_df = concept_retraction_df[
    concept_retraction_df["counts"].apply(lambda x: x["retracted"] > 10)
]

# Sort the DataFrame by the percentage of retractions in descending order
concept_retraction_df = concept_retraction_df.sort_values(
    by="percentage", ascending=False
)

# Save the results to a CSV file
concept_retraction_df.to_csv("results/concept_retraction_counts.csv", index=False)

# Convert the concept_count_by_relation_year dictionary to a DataFrame
concept_count_df = pd.DataFrame(
    list(concept_count_by_relation_year.items()),
    columns=["relation_year_concept", "count"],
)

# Extract relation, year, and concept from the 'relation_year_concept' column
concept_count_df[["relation", "year", "concept"]] = pd.DataFrame(
    concept_count_df["relation_year_concept"].tolist(), index=concept_count_df.index
)
concept_count_df = concept_count_df[
    ~concept_count_df["concept"].isin(excluded_concepts)
]

# Find the top 10 most frequent concepts for each relation and year
top_concepts_by_relation_year = (
    concept_count_df.groupby(["relation", "year"])
    .apply(lambda group: group.nlargest(10, "count"))
    .reset_index(drop=True)
)

concept_colors = {
    "Physics": "red",
    "Artificial intelligence": "blue",
    "Programming language": "green",
    "Algorithm": "orange",
    "Telecommunications": "purple",
    "Materials science": "cyan",
    "Psychology": "magenta",
    "Biology": "brown",
    "Mathematics": "pink",
    "Engineering": "gray",
    "Quantum mechanics": "lime",
    "Chemistry": "olive",
    "Operating system": "maroon",
    "Medicine": "navy",
    "Economics": "teal",
    "Mathematical analysis": "teal",
    "Optics": "teal",
}

# Plot the top 10 most frequent concepts for each relation by year
for relation in new_df["relation"].unique():
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x="year",
        y="count",
        hue="concept",
        data=top_concepts_by_relation_year[
            (top_concepts_by_relation_year["relation"] == relation)
            & (top_concepts_by_relation_year["count"] > 0)
        ],
    )
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.title(f"Top 10 Most Frequent Concepts for Relation: {relation}")
    plt.legend(title="Concept", loc="upper right")
    plt.savefig(f"results/lineplot_top_concepts_{relation}.png")
    plt.close()

# Plot the top 5 most common institution types by relation
plt.figure(figsize=(12, 6))
sns.barplot(
    x="institution_type",
    y="institution_type_count",
    hue="relation",
    data=top_institution_types_by_relation,
)
plt.xlabel("Institution Type")
plt.ylabel("Count")
plt.title("Top 5 Most Common Institution Types by Relation")
plt.legend(title="Relation", loc="upper right")
plt.savefig("results/barplot_institution_type_per_relation.png")
plt.close()"""

mixed_df = new_df[new_df["relation"] == "Other Countries"]

country_counts = {}
country_citations = {}
for row in tqdm(
    mixed_df.itertuples(),
    total=len(mixed_df),
    desc="Counting Countries for Other Countries Relation",
):
    countries = set(literal_eval(row.countries))
    for country in countries:
        if country in EU_COUNTRIES:
            country = "EU"
        elif country in CN_COUNTRIES:
            country = "CN"
        country_counts[country] = country_counts.get(country, 0) + 1
        country_citations[country] = country_citations.get(country, 0) + row.citations

# Create a DataFrame with country counts and citations
country_stats_df = pd.DataFrame(
    {
        "Country": list(country_counts.keys()),
        "Count": list(country_counts.values()),
        "Total Citations": list(country_citations.values()),
    }
)

# Calculate average citations
country_stats_df["Average Citations"] = (
    country_stats_df["Total Citations"] / country_stats_df["Count"]
)

country_stats_df = country_stats_df.sort_values(by="Average Citations", ascending=False)

# Save the whole list to CSV
country_stats_df.to_csv("results/all_countries_mixed_relation.csv", index=False)

# Select top 10 countries by count
top_countries_by_count = country_stats_df.nlargest(10, "Count")

# Plot barplot for top 10 countries by count
plt.figure(figsize=(10, 6))
plt.bar(top_countries_by_count["Country"], top_countries_by_count["Count"])
plt.xlabel("Country")
plt.ylabel("Publications")
plt.title("Top 10 Most Published Countries for Mixed Relation")
plt.xticks(rotation=45, ha="right")
plt.savefig("results/bar_plot_most_published_countries.png")
plt.close()

# Select top 10 countries by average citation
top_countries_by_citations = country_stats_df.nlargest(10, "Average Citations")

# Plot barplot for top 10 countries by average citation
plt.figure(figsize=(10, 6))
plt.bar(
    top_countries_by_citations["Country"],
    top_countries_by_citations["Average Citations"],
)
plt.xlabel("Country")
plt.ylabel("Average Citations")
plt.title("Top 10 Countries by Average Citation for Mixed Relation")
plt.xticks(rotation=45, ha="right")
plt.savefig("results/bar_plot_highest_avg_citations.png")
plt.close()

country_stats_df = country_stats_df[
    ~country_stats_df["Country"].isin(["EU", "CN", "US"])
]

# Calculate average citations
country_stats_df["Average Citations"] = (
    country_stats_df["Total Citations"] / country_stats_df["Count"]
)

country_stats_df = country_stats_df.sort_values(by="Average Citations", ascending=False)

# Save the whole list to CSV
country_stats_df.to_csv(
    "results/all_countries_mixed_relation_without_selected.csv", index=False
)

# Select top 10 countries by count
top_countries_by_count = country_stats_df.nlargest(10, "Count")

# Plot barplot for top 10 countries by count
plt.figure(figsize=(10, 6))
plt.bar(top_countries_by_count["Country"], top_countries_by_count["Count"])
plt.xlabel("Country")
plt.ylabel("Publications")
plt.title("Top 10 Most Published Countries for Mixed Relation (Excluding US, CN, EU)")
plt.xticks(rotation=45, ha="right")
plt.savefig("results/bar_plot_most_published_countries_without_selected.png")
plt.close()

# Select top 10 countries by average citation
top_countries_by_citations = country_stats_df.nlargest(10, "Average Citations")

# Plot barplot for top 10 countries by average citation
plt.figure(figsize=(10, 6))
plt.bar(
    top_countries_by_citations["Country"],
    top_countries_by_citations["Average Citations"],
)
plt.xlabel("Country")
plt.ylabel("Average Citations")
plt.title(
    "Top 10 Countries by Average Citation for Mixed Relation (Excluding US, CN, EU)"
)
plt.xticks(rotation=45, ha="right")
plt.savefig("results/bar_plot_highest_avg_citations_without_selected.png")
plt.close()


other_countries_df = new_df[new_df["relation"] == "Other Countries"]

country_counts = {}
country_citations = {}
for row in tqdm(
    other_countries_df.itertuples(),
    total=len(other_countries_df),
    desc="Counting Countries for Other Countries Relation",
):
    countries = set(literal_eval(row.countries))
    for country in countries:
        if country in EU_COUNTRIES:
            country = "EU"
        elif country in CN_COUNTRIES:
            country = "CN"
        country_counts[country] = country_counts.get(country, 0) + 1
        country_citations[country] = country_citations.get(country, 0) + row.citations

# Create a DataFrame with country counts and citations
country_stats_df = pd.DataFrame(
    {
        "Country": list(country_counts.keys()),
        "Count": list(country_counts.values()),
        "Total Citations": list(country_citations.values()),
    }
)

# Calculate average citations
country_stats_df["Average Citations"] = (
    country_stats_df["Total Citations"] / country_stats_df["Count"]
)

country_stats_df = country_stats_df.sort_values(by="Average Citations", ascending=False)

# Save the whole list to CSV
country_stats_df.to_csv(
    "results/all_countries_other_countries_relation.csv", index=False
)

# Select top 10 countries by count
top_countries_by_count = country_stats_df.nlargest(10, "Count")

# Plot barplot for top 10 countries by count
plt.figure(figsize=(10, 6))
plt.bar(top_countries_by_count["Country"], top_countries_by_count["Count"])
plt.xlabel("Country")
plt.ylabel("Publications")
plt.title("Top 10 Most Published Countries for Other Countries Relation")
plt.xticks(rotation=45, ha="right")
plt.savefig("results/bar_plot_most_published_countries_other_countries.png")
plt.close()

# Select top 10 countries by average citation
top_countries_by_citations = country_stats_df.nlargest(10, "Average Citations")

# Plot barplot for top 10 countries by average citation
plt.figure(figsize=(10, 6))
plt.bar(
    top_countries_by_citations["Country"],
    top_countries_by_citations["Average Citations"],
)
plt.xlabel("Country")
plt.ylabel("Average Citations")
plt.title("Top 10 Countries by Average Citation for Other Countries Relation")
plt.xticks(rotation=45, ha="right")
plt.savefig("results/bar_plot_highest_avg_citations_other_countries.png")
plt.close()

country_stats_df = country_stats_df[
    ~country_stats_df["Country"].isin(["EU", "CN", "US"])
]

# Calculate average citations
country_stats_df["Average Citations"] = (
    country_stats_df["Total Citations"] / country_stats_df["Count"]
)

country_stats_df = country_stats_df.sort_values(by="Average Citations", ascending=False)

# Save the whole list to CSV
country_stats_df.to_csv(
    "results/all_countries_other_countries_relation_without_selected.csv", index=False
)

# Select top 10 countries by count
top_countries_by_count = country_stats_df.nlargest(10, "Count")

# Plot barplot for top 10 countries by count
plt.figure(figsize=(10, 6))
plt.bar(top_countries_by_count["Country"], top_countries_by_count["Count"])
plt.xlabel("Country")
plt.ylabel("Publications")
plt.title(
    "Top 10 Most Published Countries for Other Countries Relation (Excluding US, CN, EU)"
)
plt.xticks(rotation=45, ha="right")
plt.savefig(
    "results/bar_plot_most_published_countries_without_selected_other_countries.png"
)
plt.close()

# Select top 10 countries by average citation
top_countries_by_citations = country_stats_df.nlargest(10, "Average Citations")

# Plot barplot for top 10 countries by average citation
plt.figure(figsize=(10, 6))
plt.bar(
    top_countries_by_citations["Country"],
    top_countries_by_citations["Average Citations"],
)
plt.xlabel("Country")
plt.ylabel("Average Citations")
plt.title(
    "Top 10 Countries by Average Citation for Other Countries Relation (Excluding US, CN, EU)"
)
plt.xticks(rotation=45, ha="right")
plt.savefig(
    "results/bar_plot_highest_avg_citations_without_selected_other_countries.png"
)
plt.close()


# Iterate through the rows and tag non-developed based on 'hdi' column
non_developed_tag = []
for row in tqdm(
    mixed_df.itertuples(),
    total=len(mixed_df),
    desc="Counting Countries for Mixed Relation",
):
    hdi_values = literal_eval(row.hdi)
    if any(value <= 0.549 for value in hdi_values):
        non_developed_tag.append(True)
    else:
        non_developed_tag.append(False)

# Add 'non_developed' column to the DataFrame
mixed_df["non_developed"] = non_developed_tag

# Separate the DataFrame into two groups based on 'non_developed' column
non_developed_group = mixed_df[mixed_df["non_developed"]]
developed_group = mixed_df[~mixed_df["non_developed"]]

# Perform statistical tests (e.g., t-test) to check for differences between the groups
statistical_test_result = stats.ttest_ind(
    non_developed_group["citations"],
    developed_group["citations"],
    equal_var=False,  # Assuming unequal variances
)

# Print the statistical test result
result_string = f"Statistical Test Result:\nT-statistic: {statistical_test_result.statistic}\nP-value: {statistical_test_result.pvalue}"

# Save the result to a text file
with open("results/non_developed_country_participating.txt", "w") as f:
    f.write(result_string)

# Plot the average citations for each group
plt.figure(figsize=(10, 6))
plt.bar(
    ["Has Low-developed", "No Low-developed"],
    [non_developed_group["citations"].mean(), developed_group["citations"].mean()],
)
plt.xlabel("Development Status")
plt.ylabel("Average Citations")
plt.title("Average Citations for Collaborations with at least a Low Developed Country")
plt.savefig("results/bar_plot_average_citations_low_developed.png")
plt.close()

# Iterate through the rows and tag high developed based on 'hdi' column
high_developed_tag = []
for row in tqdm(
    mixed_df.itertuples(),
    total=len(mixed_df),
    desc="Counting Countries for Mixed Relation",
):
    hdi_values = literal_eval(row.hdi)
    if any(value >= 0.799 for value in hdi_values):
        high_developed_tag.append(True)
    else:
        high_developed_tag.append(False)

# Add 'high_developed' column to the DataFrame
mixed_df["high_developed"] = high_developed_tag

# Separate the DataFrame into two groups based on 'high_developed' column
high_developed_group = mixed_df[mixed_df["high_developed"]]
other_group = mixed_df[~mixed_df["high_developed"]]

# Perform statistical tests (e.g., t-test) to check for differences between the groups
statistical_test_result = stats.ttest_ind(
    high_developed_group["citations"],
    other_group["citations"],
    equal_var=False,  # Assuming unequal variances
)

result_string = f"Statistical Test Result:\nT-statistic: {statistical_test_result.statistic}\nP-value: {statistical_test_result.pvalue}"

# Save the result to a text file
with open("results/developed_country_participating.txt", "w") as f:
    f.write(result_string)
# Plot the average citations for each group
plt.figure(figsize=(10, 6))
plt.bar(
    ["Has Very High Developed", "No Very High Developed"],
    [high_developed_group["citations"].mean(), other_group["citations"].mean()],
)
plt.xlabel("Development Status")
plt.ylabel("Average Citations")
plt.title(
    "Average Citations for Collaborations with at least a Very High Developed Country"
)
plt.savefig("results/bar_plot_average_citations_high_developed.png")
plt.close()

# Calculate Pearson correlation coefficient
pearson_corr, pearson_p_value = pearsonr(
    mixed_df["citations"],
    mixed_df["hdi"].apply(
        lambda x: sum(literal_eval(x)) / len(literal_eval(x))
        if len(literal_eval(x)) > 0
        else 0
    ),
)

# Calculate Spearman correlation coefficient
spearman_corr, spearman_p_value = spearmanr(
    mixed_df["citations"],
    mixed_df["hdi"].apply(
        lambda x: sum(literal_eval(x)) / len(literal_eval(x))
        if len(literal_eval(x)) > 0
        else 0
    ),
)

# Print and save the correlation coefficients
correlation_result = f"Pearson Correlation Coefficient: {pearson_corr} (p-value: {pearson_p_value})\nSpearman Correlation Coefficient: {spearman_corr} (p-value: {spearman_p_value})"

# Save the correlation result to a text file
with open("results/correlation_result.txt", "w") as f:
    f.write(correlation_result)

# Save the mean citations for each group to a text file (non-developed)
mean_citations_non_developed = {
    "Has Non-developed": non_developed_group["citations"].mean(),
    "Has Only Developed": developed_group["citations"].mean(),
}

with open("results/mean_citations_non_developed.txt", "w") as f:
    f.write(
        "Mean Citations for Collaborations with at least a Low Developed Country:\n"
    )
    for group, mean in mean_citations_non_developed.items():
        f.write(f"{group}: {mean}\n")

# Save the mean citations for each group to a text file (high-developed)
mean_citations_high_developed = {
    "Has Very High Developed": high_developed_group["citations"].mean(),
    "Without Very High Developed": other_group["citations"].mean(),
}

with open("results/mean_citations_high_developed.txt", "w") as f:
    f.write(
        "Mean Citations for Collaborations with at least a Very High Developed Country:\n"
    )
    for group, mean in mean_citations_high_developed.items():
        f.write(f"{group}: {mean}\n")

# Perform statistical tests (t-test) for non-developed group
t_statistic_non_developed, p_value_non_developed = stats.ttest_ind(
    non_developed_group["citations"],
    developed_group["citations"],
    equal_var=False,  # Assuming unequal variances
)

# Save the t-test result to a text file (non-developed)
result_string_non_developed = (
    "APA Format Result for Collaborations with at least a Low Developed Country:\n\n"
)
result_string_non_developed += f"T({len(non_developed_group['citations']) + len(developed_group['citations']) - 2:.0f}) = {t_statistic_non_developed:.2f}, p = {p_value_non_developed:.3f}\n\n"

# Interpretation
if p_value_non_developed < 0.05:
    result_string_non_developed += (
        "The difference in citations between groups is statistically significant.\n"
    )
else:
    result_string_non_developed += "There is no statistically significant difference in citations between groups.\n"

# Save the result to a text file
with open("results/t_test_non_developed_apa.txt", "w") as f:
    f.write(result_string_non_developed)

# Perform statistical tests (t-test) for high-developed group
t_statistic_high_developed, p_value_high_developed = stats.ttest_ind(
    high_developed_group["citations"],
    other_group["citations"],
    equal_var=False,  # Assuming unequal variances
)

# Save the t-test result to a text file (high-developed)
result_string_high_developed = "APA Format Result for Collaborations with at least a Very High Developed Country:\n\n"
result_string_high_developed += f"T({len(high_developed_group['citations']) + len(other_group['citations']) - 2:.0f}) = {t_statistic_high_developed:.2f}, p = {p_value_high_developed:.3f}\n\n"

# Interpretation
if p_value_high_developed < 0.05:
    result_string_high_developed += (
        "The difference in citations between groups is statistically significant.\n"
    )
else:
    result_string_high_developed += "There is no statistically significant difference in citations between groups.\n"

# Save the result to a text file
with open("results/t_test_high_developed_apa.txt", "w") as f:
    f.write(result_string_high_developed)

# Calculate for non_developed_group
num_retracted_non_developed = non_developed_group["is_retracted"].sum()
total_works_non_developed = len(non_developed_group)
percentage_retracted_non_developed = (
    num_retracted_non_developed / total_works_non_developed
) * 100
num_retracted_developed = developed_group["is_retracted"].sum()
total_works_developed = len(developed_group)
percentage_retracted_developed = (num_retracted_developed / total_works_developed) * 100


# Save the results to a text file for non_developed_group
with open("results/retracted_stats_non_developed.txt", "w") as f:
    f.write(
        f"Number of Retracted Works for Non-Developed Group: {num_retracted_non_developed}\n"
    )
    f.write(
        f"Total Number of Works for Non-Developed Group: {total_works_non_developed}\n"
    )
    f.write(
        f"Percentage of Retracted Works for Non-Developed Group: {percentage_retracted_non_developed:.2f}%\n"
    )
    f.write(
        f"Number of Retracted Works for Developed Group: {num_retracted_developed}\n"
    )
    f.write(f"Total Number of Works for Developed Group: {total_works_developed}\n")
    f.write(
        f"Percentage of Retracted Works for Developed Group: {percentage_retracted_developed:.2f}%\n"
    )

# Calculate for high_developed_group
num_retracted_high_developed = high_developed_group["is_retracted"].sum()
total_works_high_developed = len(high_developed_group)
percentage_retracted_high_developed = (
    num_retracted_high_developed / total_works_high_developed
) * 100
num_retracted_other = other_group["is_retracted"].sum()
total_works_other = len(other_group)
percentage_retracted_other = (num_retracted_other / total_works_other) * 100

# Save the results to a text file for high_developed_group
with open("results/retracted_stats_high_developed.txt", "w") as f:
    f.write(
        f"Number of Retracted Works for High-Developed Group: {num_retracted_high_developed}\n"
    )
    f.write(
        f"Total Number of Works for High-Developed Group: {total_works_high_developed}\n"
    )
    f.write(
        f"Percentage of Retracted Works for High-Developed Group: {percentage_retracted_high_developed:.2f}%\n"
    )
    f.write(f"Number of Retracted Works for Other Group: {num_retracted_other}\n")
    f.write(f"Total Number of Works for Other Group: {total_works_other}\n")
    f.write(
        f"Percentage of Retracted Works for Other Group: {percentage_retracted_other:.2f}%\n"
    )

"""# Create a regression plot
plt.figure(figsize=(10, 6))
sns.regplot(
    x=mixed_df["citations"],
    y=mixed_df["hdi"].apply(
        lambda x: sum(literal_eval(x)) / len(literal_eval(x))
        if len(literal_eval(x)) > 0
        else 0
    ),
    scatter_kws={"s": 50},
)
plt.ylabel("Mean Citations")
plt.xlabel("Mean HDI")
plt.title("Regression Plot between Mean Citations and Mean HDI")
plt.savefig("results/regression_plot_mean_citations_mean_hdi.png")
plt.close()
"""

concept_colors = {
    "Physics": "red",
    "Artificial intelligence": "blue",
    "Programming language": "green",
    "Machine learning": "orange",
    "Optics": "purple",
    "Materials science": "cyan",
    "Psychology": "magenta",
    "Biology": "brown",
    "Mathematics": "pink",
    "Engineering": "gray",
    "Quantum mechanics": "lime",
    "Chemistry": "olive",
    "Operating system": "maroon",
    "Medicine": "navy",
    "Economics": "teal",
}

us_eu_cn_rows = {}
mixed_rows = {}
other_rows = {}

for row in tqdm(
    new_df.itertuples(),
    total=len(new_df),
    desc="Counting Institution Types and Concepts",
):
    concepts = set(literal_eval(row.concepts))
    relation = str(row.relation)
    for concept in concepts:
        if relation == "US-EU-CN":
            us_eu_cn_rows[concept] = us_eu_cn_rows.get(concept, 0) + 1
        elif relation == "Mixed":
            mixed_rows[concept] = mixed_rows.get(concept, 0) + 1
        elif relation == "Other Countries":
            other_rows[concept] = other_rows.get(concept, 0) + 1

# Convert dictionaries to DataFrames
us_eu_cn_df = pd.DataFrame(list(us_eu_cn_rows.items()), columns=["concept", "count"])
mixed_df = pd.DataFrame(list(mixed_rows.items()), columns=["concept", "count"])
other_df = pd.DataFrame(list(other_rows.items()), columns=["concept", "count"])


# Function to plot top N concepts for a DataFrame and save the plot
def plot_top_concepts_and_save(temp_df, title):
    temp_df = temp_df[temp_df["concept"] != "Computer science"]
    df_sorted = temp_df.sort_values(by="count", ascending=False)
    top_10 = df_sorted.head(10)
    sns.barplot(top_10["concept"], top_10["count"], palette=concept_colors)
    plt.xlabel("Concept")
    plt.ylabel("Number of occurrences")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.savefig(
        f"results/{title.lower().replace(' ', '_')}_top_concepts.png",
        bbox_inches="tight",
    )
    plt.close()


# Plot top 10 concepts for each DataFrame and save the plots
plot_top_concepts_and_save(
    us_eu_cn_df, "Most frequent Concepts in US-EU-CN colaborations"
)
plot_top_concepts_and_save(mixed_df, "Most frequent Concepts in Mixed colaborations")
plot_top_concepts_and_save(
    other_df, "Most frequent Concepts in Other Countries colaborations"
)

us_eu_cn_rows = []
mixed_rows = []
other_rows = []

for row in tqdm(
    new_df.itertuples(),
    total=len(new_df),
    desc="Counting Institution Types and Concepts",
):
    concepts = set(literal_eval(row.concepts))
    relation = str(row.relation)
    year = row.publication_year
    for concept in concepts:
        if relation == "US-EU-CN":
            us_eu_cn_rows.append((year, concept))
        elif relation == "Mixed":
            mixed_rows.append((year, concept))
        elif relation == "Other Countries":
            other_rows.append((year, concept))

# Create DataFrames for each relation
us_eu_cn_df = pd.DataFrame(us_eu_cn_rows, columns=["year", "concept"])
mixed_df = pd.DataFrame(mixed_rows, columns=["year", "concept"])
other_df = pd.DataFrame(other_rows, columns=["year", "concept"])


def plot_top_concepts_by_year(temp_df: pd.DataFrame, title):
    temp_df = temp_df[temp_df["concept"] != "Computer science"]
    df_grouped = temp_df.groupby(["year", "concept"]).size().reset_index(name="count")
    top_concepts_2021 = df_grouped[df_grouped["year"] == 2021].nlargest(10, "count")
    top_concepts_2021.to_csv("results/top_concepts_2021.txt", sep="\t", index=False)
    top_concepts_by_year = df_grouped[
        df_grouped["concept"].isin(top_concepts_2021["concept"])
    ]
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x="year",
        y="count",
        hue="concept",
        data=top_concepts_by_year,
        palette=concept_colors,
    )
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Number of occurrences")
    plt.legend(title="Concept", loc="upper left")
    plt.savefig(
        f"results/{title.lower().replace(' ', '_')}_top_concepts_by_year.png",
        bbox_inches="tight",
    )
    plt.close()


# Plot top 10 concepts by year for each DataFrame
plot_top_concepts_by_year(
    us_eu_cn_df, "Most frequent Concepts in US-EU-CN colaborations by Year"
)
plot_top_concepts_by_year(
    mixed_df, "Most frequent Concepts in Mixed colaborations by Year"
)
plot_top_concepts_by_year(
    other_df, "Most frequent Concepts in Other Countries colaborations by Year"
)
