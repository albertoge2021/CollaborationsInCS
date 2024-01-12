from ast import literal_eval
from collections import Counter, defaultdict
import json
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


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

# Setup Data
df = pd.read_csv("data_countries/cs_works.csv")

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

total_collaborations = defaultdict(dict)

for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    country_list = literal_eval(row.countries)
    country_list = [
        "RS" if country == "XK" else "CN" if country == "TW" else country
        for country in country_list
    ]
    country_list = list(set(country_list))
    if len(country_list) <= 1:
        continue

    # Iterate through each combination of two countries in the current row
    for i in range(len(country_list)):
        for j in range(i + 1, len(country_list)):
            country1, country2 = country_list[i], country_list[j]
            # Exclude self-collaborations and handle None values
            if country1 is not None and country2 is not None and country1 != country2:
                # Update total_collaborations for country1
                total_collaborations[country1][country2] = (
                    total_collaborations[country1].get(country2, 0) + 1
                )

                # Update total_collaborations for country2
                total_collaborations[country2][country1] = (
                    total_collaborations[country2].get(country1, 0) + 1
                )

# Sort the total_collaborations dictionary by keys and create a new sorted dictionary
sorted_collaborations = {
    country: dict(sorted(collabs.items()))
    for country, collabs in total_collaborations.items()
}
with open("sorted_collaborations.json", "w") as json_file:
    json.dump(sorted_collaborations, json_file)

with open("sorted_collaborations.json", "r") as json_file:
    sorted_collaborations = json.load(json_file)
threshold_percentage = 1  # You can adjust this threshold as needed

for collaborator, collaborations in sorted_collaborations.items():
    collabs = []
    no_collabs = []

    total_collaborations = sum(collaborations.values())

    for country, no_collaborations in collaborations.items():
        percentage = (no_collaborations / total_collaborations) * 100
        if percentage >= threshold_percentage:
            collabs.append(
                pc.country_name_to_country_alpha3(
                    pc.country_alpha2_to_country_name(country)
                )
            )
            no_collabs.append(no_collaborations)

    # Check if there are collaborations below the threshold
    if total_collaborations - sum(no_collabs) > 0:
        collabs.append("Others")
        no_collabs.append(total_collaborations - sum(no_collabs))

    country_df = pd.DataFrame({"collaborations": no_collabs, "country": collabs})

    # plot it
    squarify.plot(
        sizes=country_df["collaborations"],
        label=country_df["country"],
        alpha=0.8,
        text_kwargs={"fontsize": 8, "wrap": True},
    )
    plt.title(f"Collaborations of {pc.country_alpha2_to_country_name(collaborator)}")
    plt.axis("off")
    plt.savefig(f"paper_results_2/tree_maps/{collaborator}_treemap.png")
    plt.close()
total_collaborations_by_country = defaultdict(int)

# Iterate through the sorted_collaborations dictionary and accumulate the counts
for country, collaborations in sorted_collaborations.items():
    total_collaborations = sum(collaborations.values())
    total_collaborations_by_country[country] = total_collaborations

# Convert the defaultdict to a regular dictionary if needed
total_collaborations_by_country = dict(total_collaborations_by_country)
collaborations_df = pd.DataFrame(
    total_collaborations_by_country.items(), columns=["Country", "Collaborations"]
)

eng_df = pd.read_csv("english_level.csv")
# Merge 'eng_df' and 'collaborations_df' by the "Country" column
merged_df = pd.merge(
    eng_df[["Country", "Score"]], collaborations_df, on="Country", how="left"
)

# Drop rows with missing or infinite values
merged_df = merged_df.dropna()
merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()

# Create a scatter plot to visualize the relationship between Score and Collaborations
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
corr_coeff_spear, p_value_spear = spearmanr(
    merged_df["Score"], merged_df["Collaborations"]
)
with open("paper_results_2/english_level_results.txt", "w") as file:
    file.write(
        f"Pearson Correlation Coefficient: {corr_coeff:.2f}"
        + f" - P-Value: {p_value:.5f}\n"
    )
    file.write(
        f"Spearman Correlation Coefficient: {corr_coeff_spear:.2f}"
        + f" - P-Value: {p_value_spear:.5f}"
    )

collaborations_by_year_and_country = defaultdict(lambda: defaultdict(dict))

# Iterate through the data and accumulate collaborations by publication_year, country, and other country
for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    publication_year = row.publication_year
    country_list = literal_eval(row.countries)
    country_list = [
        "RS" if country == "XK" else "CN" if country == "TW" else country
        for country in country_list
    ]
    country_list = list(set(country_list))
    if len(country_list) <= 1:
        continue

    # Iterate through each combination of two countries in the current row
    for i in range(len(country_list)):
        for j in range(i + 1, len(country_list)):
            country1, country2 = country_list[i], country_list[j]

            # Exclude self-collaborations and handle None values
            if country1 is not None and country2 is not None and country1 != country2:
                # Update collaborations_by_year_and_country for country1
                collaborations_by_year_and_country[publication_year][country1][
                    country2
                ] = (
                    collaborations_by_year_and_country[publication_year][country1].get(
                        country2, 0
                    )
                    + 1
                )

                # Update collaborations_by_year_and_country for country2
                collaborations_by_year_and_country[publication_year][country2][
                    country1
                ] = (
                    collaborations_by_year_and_country[publication_year][country2].get(
                        country1, 0
                    )
                    + 1
                )

# Convert the defaultdict to a regular dictionary if needed
collaborations_by_year_and_country = {
    publication_year: {country: dict(data) for country, data in data_dict.items()}
    for publication_year, data_dict in collaborations_by_year_and_country.items()
}
total_collaborations_by_year_and_country = defaultdict(lambda: defaultdict(int))

# Iterate through collaborations_by_year_and_country to calculate total collaborations
for publication_year, year_data in collaborations_by_year_and_country.items():
    for country1, country_data in year_data.items():
        total_collaborations = sum(country_data.values())
        total_collaborations_by_year_and_country[publication_year][
            country1
        ] = total_collaborations

# Convert the defaultdict to a regular dictionary if needed
total_collaborations_by_year_and_country = {
    publication_year: dict(data)
    for publication_year, data in total_collaborations_by_year_and_country.items()
}

data_list = []
for publication_year, year_data in total_collaborations_by_year_and_country.items():
    for country, total_collaborations in year_data.items():
        data_list.append(
            {
                "Year": publication_year,
                "Country": country,
                "TotalCollaborations": total_collaborations,
            }
        )

# Create a DataFrame from the list of dictionaries
total_collaborations_df = pd.DataFrame(data_list)

merged_data = pd.merge(
    total_collaborations_df,
    dev_df,
    left_on=["Year", "Country"],
    right_on=["Year", "Code"],
    how="left",
)

# Drop rows with missing values in the 'Hdi' column
merged_data = merged_data.dropna(subset=["Hdi"])


plt.scatter(merged_data["Hdi"], merged_data["TotalCollaborations"], alpha=0.5)
plt.title("Total Collaborations vs. HDI")
plt.xlabel("HDI")
plt.ylabel("Total Collaborations")
plt.grid(True)
plt.savefig(f"paper_results_2/development/scatter_hdi_publications.png")
plt.close()

# Select the features for clustering
X = merged_data[["Hdi", "TotalCollaborations"]]

# Define the number of clusters (you can adjust this)
k = 3
kmeans = KMeans(n_clusters=k)
merged_data["Cluster"] = kmeans.fit_predict(X)

# Plot the clustered data
plt.figure(figsize=(10, 6))
for i in range(k):
    cluster_data = merged_data[merged_data["Cluster"] == i]
    plt.scatter(
        cluster_data["Hdi"], cluster_data["TotalCollaborations"], label=f"Cluster {i}"
    )

plt.title("Total Collaborations vs. HDI (Clustered)")
plt.xlabel("HDI")
plt.ylabel("Total Collaborations")
plt.legend()
plt.grid(True)
plt.savefig(f"paper_results_2/development/scatter_hdi_publications_cluster.png")
plt.close()

threshold = 0.8  # Adjust this value to your threshold of interest
plt.figure(figsize=(10, 6))
plt.scatter(merged_data["Hdi"], merged_data["TotalCollaborations"], alpha=0.5)
plt.axvline(
    x=threshold, color="r", linestyle="--", label=f"HDI Threshold ({threshold})"
)
plt.title("Total Collaborations vs. HDI")
plt.xlabel("HDI")
plt.ylabel("Total Collaborations")
plt.legend()
plt.grid(True)
plt.savefig(f"paper_results_2/development/scatter_hdi_publications_threshold.png")
plt.close()

# 4. HDI Grouping
# Create HDI groups
bins = [0, 0.549, 0.699, 0.799, 1.0]
labels = ["Low", "Medium", "High", "Very High"]
merged_data["HdiGroup"] = pd.cut(merged_data["Hdi"], bins=bins, labels=labels)

# Calculate average/median total collaborations by HDI group
average_collaborations_by_hdi = merged_data.groupby("HdiGroup")[
    "TotalCollaborations"
].mean()
median_collaborations_by_hdi = merged_data.groupby("HdiGroup")[
    "TotalCollaborations"
].median()
with open("paper_results_2/development/hdi_results.txt", "a") as file:
    file.write("Average Total Collaborations by HDI Group:\n")
    file.write(average_collaborations_by_hdi.to_string())
    file.write("\n")
    file.write("Median Total Collaborations by HDI Group:\n")
    file.write(median_collaborations_by_hdi.to_string())
    file.write("\n")

corr_coeff, p_value = pearsonr(merged_data["Hdi"], merged_data["TotalCollaborations"])
corr_coeff_spear, p_value_spear = spearmanr(
    merged_data["Hdi"], merged_data["TotalCollaborations"]
)
with open("paper_results_2/development/hdi_results.txt", "a") as file:
    file.write(
        f"Pearson Correlation Coefficient: {corr_coeff:.2f}"
        + f" - P-Value: {p_value:.5f}\n"
    )
    file.write(
        f"Spearman Correlation Coefficient: {corr_coeff_spear:.2f}"
        + f" - P-Value: {p_value_spear:.5f}\n"
    )

"""df_concepts_country = pd.DataFrame(data_country, columns=['concept', 'publication_year', 'citations', 'type', 'country', 'max_distance', 'avg_distance', 'has_no_dev_country'])

selected_countries = ['KR', 'DE', 'US', 'CN', 'JP', 'GB', 'CA', 'AU', 'FR', 'ID', 'IN', 'BR', 'RU']
selected_concept = 'Artificial intelligence'
filtered_df = df_concepts_country[(df_concepts_country['country'].isin(selected_countries)) & (df_concepts_country['concept'] == selected_concept)]

# Count the number of papers (rows) for each publication_year and country
paper_counts = filtered_df.groupby(['publication_year', 'country']).size().reset_index(name='paper_count')

# Create a lineplot using seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(data=paper_counts, x='publication_year', y='paper_count', hue='country')
plt.title(f'Number of Papers for "{selected_concept}" by Year')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.legend(title='Country', loc='upper left')
plt.savefig(f"paper_results_2/line_topics_by_year_{selected_concept}.png")
plt.close()"""

def has_no_dev_country(hdi_list):
    return any(float(x) < 0.549 for x in literal_eval(hdi_list))

# Apply the function to create the new column
df['has_no_dev_country'] = df['hdi'].apply(has_no_dev_country)
# Task 1: Calculate and save statistics
grouped_stats = (
    df.groupby("has_no_dev_country")
    .agg(
        {
            "citations": ["mean", "median"],
            # "max_distance": ["mean", "median"],
            # "avg_distance": ["mean", "median"],
        }
    )
    .reset_index()
)

grouped_stats.columns = [
    "has_no_dev_country",
    "Avg_Citations",
    "Median_Citations",
    # "Avg_Max_Distance",
    # "Median_Max_Distance",
    # "Avg_Avg_Distance",
    # "Median_Avg_Distance",
]

grouped_stats.to_csv(
    "paper_results_2/development/grouped_statistics_concepts.csv", index=False
)

group_1 = df[df["has_no_dev_country"] == True]
group_2 = df[df["has_no_dev_country"] == False]

# Perform a two-sample t-test for 'citations' between the two groups
t_statistic, p_value = ttest_ind(
    group_1["citations"], group_2["citations"], equal_var=False
)

with open("paper_results_2/development/hdi_results.txt", "a") as file:
    file.write(f"T-Test results: {t_statistic:.2f}" + f" - P-Value: {p_value:.5f}\n")
    alpha = 0.05
    if p_value < alpha:
        file.write(
            "There is a statistically significant difference between the groups.\n"
        )
    else:
        file.write(
            "There is no statistically significant difference between the groups.\n"
        )

data = []
data_country = []
# Initialize an empty graph
G = nx.Graph()

for row in tqdm(
    df.itertuples(), total=len(df), desc="Counting Countries"
):
    year = row.publication_year
    country_list = literal_eval(row.countries)
    country_list = [
        "RS" if country == "XK" else "CN" if country == "TW" else country
        for country in country_list
    ]
    country_list = list(set(country_list))
    if not len(country_list) > 1:
        continue
    for concept in literal_eval(row.concepts):
        data.append(
            (
                concept,
                year,
                row.citations,
                row.type,
                row.countries,
                row.has_no_dev_country,
            )
        )
        #for country in country_list:
        #    data_country.append((concept, year, row.citations, row.type, country, row.max_distance, row.avg_distance, row.has_no_dev_country))


df_concepts = pd.DataFrame(
    data,
    columns=[
        "concept",
        "year",
        "citations",
        "type",
        "countries",
        "has_no_dev_country",
    ],
)
df_concepts_country = pd.DataFrame(data_country, columns=['concept', 'year', 'citations', 'type', 'country', 'has_no_dev_country'])

for relation in df_concepts_country["has_no_dev_country"].unique():
    relation_df = df_concepts_country[df_concepts_country["has_no_dev_country"] == relation]
    test = (
        df_concepts_country.groupby("concept")
        .size()
        .reset_index(name="count")
        .sort_values(by=["count"], ascending=False)
        .head(11)
    )
    test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
    new_df = relation_df.loc[relation_df["concept"].isin(test.concept.to_list())]
    means_full = (
        new_df.groupby(["concept", "publication_year"]).size().reset_index(name="count")
    )
    sns.lineplot(
        data=means_full,
        x="publication_year",
        y="count",
        hue="concept",
        markers=True,
        sort=True,
    )
    if relation == False:
        name = "Developed"
    else:
        name = "Non-Developed"
    plt.xlabel("Year")
    plt.legend(title="Concept")
    plt.ylabel("Number of collaborations")
    plt.title("10 most common topics by publication_year for " + name)
    plt.savefig(f"paper_results_2/development/line_topics_by_year_{name}.png")
    plt.close()

# Load your data
dev_df_north_south = pd.read_csv("dev_df_north_south.csv")

# Group by 'hemisphere' column and count the number of entries in each group
grouped = dev_df_north_south.groupby("hemisphere").count()

# Calculate average and median citations for each group
avg_citations = dev_df_north_south.groupby("hemisphere")["citations"].mean()
median_citations = dev_df_north_south.groupby("hemisphere")["citations"].median()

# Perform Kruskal-Wallis test
h_statistic, p_value = stats.kruskal(
    *[group["citations"] for _, group in dev_df_north_south.groupby("hemisphere")]
)

# Open the file in append mode and write the results
with open("paper_results_2/development/hemisphere_results.txt", "w") as file:
    file.write("Hemisphere Counts:\n")
    file.write(grouped.to_string() + "\n\n")

    file.write("Average Citations by Hemisphere:\n")
    file.write(avg_citations.to_string() + "\n\n")

    file.write("Median Citations by Hemisphere:\n")
    file.write(median_citations.to_string() + "\n\n")

    file.write(f"Kruskal Test between Hemispheres:\n")
    file.write(f"T-statistic: {h_statistic}\n")
    file.write(f"P-value: {p_value}\n\n")

    if p_value < 0.05:  # You can set your significance level here
        file.write("Statistically significant differences exist between the groups.\n")
    else:
        file.write(
            "No statistically significant differences exist between the groups.\n"
        )

# Group the DataFrame by 'hemisphere' and count the number of rows in each group
hemisphere_counts = dev_df_north_south["hemisphere"].value_counts()

# Specify custom legend names
custom_legend_names = ["Northern Hemisphere", "Southern Hemisphere", "Both Hemispheres"]

# Create a pie plot
plt.figure(figsize=(8, 8))
plt.pie(
    hemisphere_counts, labels=hemisphere_counts.index, autopct="%1.1f%%", startangle=140
)
plt.title("Number of Rows by Hemisphere")
plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

# Add a legend with custom names
plt.legend(custom_legend_names, loc="upper right")

plt.savefig(f"paper_results_2/development/pie_chart_papers_by_hemisphere.png")
plt.close()

unique_hemisphere_types = dev_df_north_south["hemisphere"].unique()

for hemis in unique_hemisphere_types:
    sub_df_southern = dev_df_north_south[dev_df_north_south["hemisphere"] == hemis]

    # Extract the "countries" column as a list of lists
    countries_lists = sub_df_southern["countries"]

    # Flatten the list of lists into a single list of countries
    all_countries = [
        country
        for countries_list in countries_lists
        for country in literal_eval(countries_list)
    ]

    # Use Counter to count the occurrences of each country
    country_counts = Counter(all_countries)

    # Sort the country counts in descending order by count
    sorted_country_counts = dict(
        sorted(country_counts.items(), key=lambda x: x[1], reverse=True)
    )

    # Print the sorted counts for each country
    with open(f"paper_results_2/development/hemisphere_count_{hemis}.txt", "w") as file:
        for country, count in sorted_country_counts.items():
            file.write(f"{country}: {count}\n")


new_dev_df = dev_df.groupby(["Code_2", "Entity"])["Hdi"].mean().reset_index()
low_hdi_df = new_dev_df[new_dev_df["Hdi"] < 0.549]
df1_countries = low_hdi_df["Code_2"].unique()
df1_countries_set = set(df1_countries)
filtered = []
for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    concepts_list = literal_eval(row.concepts)
    if any(country in df1_countries_set for country in literal_eval(row.countries)):
        for concept in literal_eval(row.concepts):
            filtered.append(
                (
                    concept,
                    row.publication_year,
                    row.citations,
                    row.type,
                    literal_eval(row.countries),
                )
            )
filtered_df2 = pd.DataFrame(
    filtered,
    columns=["concept", "publication_year", "citations", "type", "countries"],
)
print(filtered_df2.head())

year_counts = filtered_df2["publication_year"].value_counts().sort_index()

# Plot the number of rows by publication_year
plt.plot(year_counts.index, year_counts.values, marker="o")
plt.xlabel("Year")
plt.ylabel("Number of Collaborations")
plt.title("Number of Collaborations by Year")
plt.savefig(
    f"paper_results_2/development/linear_plot_publications_by_year_non_developed.png"
)
plt.close()

test = (
    filtered_df2.groupby("concept")
    .size()
    .reset_index(name="count")
    .sort_values(by=["count"], ascending=False)
    .head(11)
)
test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
new_df = filtered_df2.loc[filtered_df2["concept"].isin(test.concept.to_list())]
means_full = (
    new_df.groupby(["concept", "publication_year"]).size().reset_index(name="count")
)

# Plot the data using specified colors
sns.lineplot(
    data=means_full,
    x="publication_year",
    y="count",
    hue="concept",
    markers=True,
    sort=True,
)
plt.xlabel("Year")
plt.legend(title="Concept")
plt.ylabel("Number of collaborations")
plt.title("10 most common topics by publication_year")
plt.savefig(f"paper_results_2/development/line_topics_by_year_non_developed.png")
plt.close()

citations_stats = filtered_df2["citations"].describe()

# Save the names of the countries from df1 to a file
with open("paper_results_2/development/country_names.txt", "w") as file:
    for country in df1_countries:
        file.write(f"{country}\n")

with open("paper_results_2/development/citations_stats.txt", "w") as file:
    file.write(f"{citations_stats}\n")

# Optionally, you can save the statistical analysis results to a file
citations_stats.to_csv(
    "paper_results_2/development/citations_statistics.csv", header=True
)


# ordenadar datos segun geografia, desarrollo, pib, nivel de ingles
