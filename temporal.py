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

# df = pd.read_csv("results/all_countries_other_countries_relation.csv")

selected_countries = ["US", "CN", "EU"]
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

"""# Load world map
world_map = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

for index, row in tqdm(df.iterrows(), total=len(df)):
    country = str(row['Country'])
    if country is None or country == "nan":
        continue

    # Update country codes
    if country == "XK":
        country = "RS"
    elif country == "TW":
        country = "CN"
    elif country == "HK":
        country = "CN"

    # Get alpha-3 code from pycountry
    try:
        alpha_3 = pycountry.countries.get(alpha_2=country).alpha_3
    except AttributeError:
        # Handle the case where the country code is not found
        alpha_3 = None  # You can choose a default value or handle it accordingly

    # Update the DataFrame
    df.at[index, 'Country'] = alpha_3

world_map = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
merged_df = world_map.merge(
    df, left_on="iso_a3", right_on="Country", how="left"
)
fig, ax = plt.subplots(figsize=(12, 8))
merged_df.plot(
    column="Count",
    cmap="YlOrRd",
    linewidth=0.8,
    ax=ax,
    edgecolor="0.8",
    legend=True,
)
merged_df[merged_df["iso_a3"].isin(collaborators_to_remove)].plot(
    color="Blue", ax=ax
)
ax.set_title(f"Number of collaborations between Other Countries")
plt.savefig(f"results/map_number_of_collaborations_other_countries.png")
plt.show()
plt.close()"""
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

new_df = pd.read_csv("new_df.csv")
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
    sns.set(style="whitegrid", palette=concept_colors.values())
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.barplot(x=top_10["concept"], y=top_10["count"])
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
"""

import pandas as pd
import pycountry

# Read the first DataFrame
df_without_collabs = pd.read_csv("results/all_countries_other_countries_relation.csv")

# Read the second DataFrame
df_with_collabs = pd.read_csv("results/all_countries_mixed_relation.csv")


# Function to convert ISO2 country code to country name
def iso2_to_country_name(iso2_code):
    try:
        country = pycountry.countries.get(alpha_2=str(iso2_code))
        return country.name
    except AttributeError:
        return iso2_code


# Apply the function to convert ISO2 country codes to country names
df_with_collabs["Country"] = df_with_collabs["Country"].apply(iso2_to_country_name)
df_without_collabs["Country"] = df_without_collabs["Country"].apply(
    iso2_to_country_name
)

# Rename columns
df_with_collabs = df_with_collabs.rename(
    columns={
        "Count": "Number of publications",
        "Average Citations": "Average citations",
    }
)
df_without_collabs = df_without_collabs.rename(
    columns={
        "Count": "Number of publications",
        "Average Citations": "Average citations",
    }
)

# Merge the two DataFrames on the 'Country' column
merged_df = pd.merge(
    df_with_collabs,
    df_without_collabs,
    on="Country",
    suffixes=("_with_collabs", "_without_collabs"),
)

# Sort by 'Number of publications' without collaborations with CN-EU-US
merged_df = merged_df.sort_values(
    by="Number of publications_with_collabs", ascending=False
)

# Create a new DataFrame with desired columns
final_df = merged_df[
    [
        "Country",
        "Number of publications_with_collabs",
        "Number of publications_without_collabs",
        "Average citations_with_collabs",
        "Average citations_without_collabs",
    ]
]

# Save DataFrame to LaTeX format with legend
with open("results/combined_table.tex", "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\n")
    f.write("Country & [1] & [2] & [3] & [4]  \\\n")
    f.write("\\hline\n")
    for _, row in final_df.iterrows():
        f.write(
            "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\\\n".format(
                row["Country"],
                row["Number of publications_with_collabs"],
                row["Number of publications_without_collabs"],
                row["Average citations_with_collabs"],
                row["Average citations_without_collabs"],
            )
        )
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write(
        "\caption{Legend: [1] = Number of publications with CN-EU-US, [2] = Number of publications without CN-EU-US, [3] = Average citations with CN-EU-US, [4] = Average citations without CN-EU-US}\n"
    )
    f.write("\\end{table}\n")
