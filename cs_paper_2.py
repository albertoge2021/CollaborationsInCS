from ast import literal_eval
import csv
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import pycountry


warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

# Setup Data
file_path = "data_countries/cs_works.csv"
df = pd.read_csv(file_path)

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

# region maps

collaborators = []

for row in tqdm(df.itertuples()):
    country_list = set(literal_eval(row.countries))
    if len(country_list) < 1:
        continue
    updated_list = []
    for country in country_list:
        if country is None:
            continue
        country = "RS" if country == "XK" else country
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
    plt.savefig(f"paper_results_2/map_number_of_collaborations_{origin}.png")
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
plt.savefig(f"paper_results_2/map_number_of_collaborations.png")
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
        vmin=0,
        vmax=40,
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
    plt.savefig(f"paper_results_2/map_percentage_of_collaborations_{origin}.png")
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
    vmin=0,
    vmax=40,
)
merged_df[merged_df["iso_a3"].isin(collaborators_to_remove)].plot(
    color="mediumpurple", ax=ax
)
ax.set_title(f"Percentage of collaborations")
plt.savefig(f"paper_results_2/map_percentage_of_collaborations.png")
plt.close()

# endregion

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
    country_codes = []
    for country in country_list:
        country = "RS" if country == "XK" else country
        if country in EU_COUNTRIES:
            country_codes.append("EU")
        else:
            country_codes.append(country)
    occurence_list.extend(
        (country_code, row.publication_year) for country_code in country_codes
    )

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

countries_ratio = []

for row in tqdm(df.itertuples()):
    country_list = literal_eval(row.countries)
    updated_list = []
    for country in country_list:
        country = "RS" if country == "XK" else country
        country = "EU" if country in EU_COUNTRIES else country
        updated_list.append(country)
    citations = int(row.citations)
    num_countries = len(country_list)
    selected_countries_counts = 0
    if any(country_code in updated_list for country_code in selected_countries):
        selected_countries_counts = (
            updated_list.count("US")
            + updated_list.count("CN")
            + updated_list.count("EU")
        )
        countries_ratio.append(
            (
                (selected_countries_counts / num_countries) * 100,
                citations,
                row.type,
                row.publication_year,
                "CN-US-EU",
            )
        )
    else:
        countries_ratio.append(
            (
                100,
                citations,
                row.type,
                row.publication_year,
                "Rest of the world",
            )
        )

ratio_df = pd.DataFrame(
    countries_ratio,
    columns=[
        "ratio",
        "citations",
        "type",
        "year",
        "relation",
    ],
)
means = ratio_df.groupby(["relation", "year"]).size().reset_index(name="count")
sns.lineplot(data=means, x="year", y="count", hue="relation")
plt.xlabel("Year")
plt.ylabel("Number of collaborations")
plt.title("Number of collaborations by regions per year")
plt.savefig(f"paper_results_2/lineplot_collaborations_per_year_per_collaboration.png")
plt.close()

selected_countries_ratio = ratio_df[ratio_df["relation"] == "CN-US-EU"]
sns.lmplot(
    x="ratio",
    y="citations",
    hue="type",
    data=selected_countries_ratio,
    scatter=False,
)
plt.xlabel("Participation ratio")
plt.ylabel("Number of Citations")
plt.title("CN-US-EU participation ratio vs citations")
plt.savefig(f"paper_results_2/scatter_ratio_citations_by_type_selected_countries.png")
plt.close()

sns.lmplot(
    x="ratio",
    y="citations",
    data=selected_countries_ratio,
    scatter=False,
)
plt.xlabel("Participation ratio")
plt.ylabel("Number of Citations")
plt.title("CN-US-EU participation ratio vs citations")
plt.savefig(f"paper_results_2/scatter_ratio_citations_selected_countries.png")
plt.close()


# Group the DataFrame by "relation" and calculate the mean of "citations" for each group
mean_citations = ratio_df.groupby("relation")["citations"].mean()
mean_citations.plot(kind="bar")
plt.xlabel("Relation")
plt.ylabel("Mean Citations")
plt.title("Mean Citations by Relation")
plt.savefig(f"paper_results_2/bar_mean_citations_by_countries.png")
plt.close()

for collaboration_type in unique_collaboration_types:
    collaboration_type_df = ratio_df[ratio_df["type"] == collaboration_type]
    mean_citations = collaboration_type_df.groupby("relation")["citations"].mean()
    mean_citations.plot(kind="bar")
    plt.xlabel("Relation")
    plt.ylabel("Mean Citations")
    plt.title(f"Mean Citations by Relation for {collaboration_type}")
    plt.savefig(
        f"paper_results_2/bar_mean_citations_by_countries_by_type{collaboration_type}.png"
    )
    plt.close()
    means = (
        collaboration_type_df.groupby(["relation", "year"])
        .size()
        .reset_index(name="count")
    )
    sns.lineplot(data=means, x="year", y="count", hue="relation")
    plt.xlabel("Year")
    plt.ylabel("Number of collaborations")
    plt.title("Number of collaborations by regions per year")
    plt.savefig(
        f"paper_results_2/lineplot_collaborations_per_year_per_collaboration_by_type{collaboration_type}.png"
    )
    plt.close()
