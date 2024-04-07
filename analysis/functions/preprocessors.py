import pandas as pd
import os
from tqdm import tqdm
from ast import literal_eval

EU_COUNTRIES = set(
    [
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
)
CN_COUNTRIES = set(["CN", "HK", "MO", "TW"])


def process_row(row, EU_COUNTRIES, CN_COUNTRIES):
    country_list = set(row["countries"])

    if row["type"] != "article":
        return None

    if all(country == "US" for country in country_list):
        return "US-only"
    if all(country in CN_COUNTRIES for country in country_list):
        return "CN-only"
    if all(country in EU_COUNTRIES for country in country_list):
        return "EU-only"
    if all(
        country not in EU_COUNTRIES and country not in CN_COUNTRIES and country != "US"
        for country in country_list
    ):
        return "Other countries"
    if all(
        country in EU_COUNTRIES or country in CN_COUNTRIES or country == "US"
        for country in country_list
    ):
        contains_us = "US" in country_list
        contains_cn = any(country in CN_COUNTRIES for country in country_list)
        contains_eu = any(country in EU_COUNTRIES for country in country_list)
        if contains_us and contains_cn and contains_eu:
            return "CN-EU-US"
        elif contains_us and contains_eu:
            return "EU-US"
        elif contains_us and contains_cn:
            return "CN-US"
        elif contains_cn and contains_eu:
            return "CN-EU"
    return "Mixed"


# Function to determine institution type
def get_institution_type(institution_types):
    if "company" in institution_types and "education" not in institution_types:
        return "company"
    elif "education" in institution_types and "company" not in institution_types:
        return "education"
    else:
        return "mixed"


# Function to get HDI values
def get_hdi(row, hdi_dict):
    hdi_values = []
    for country_code in row["countries"]:
        publication_year = row["publication_year"]
        if publication_year in hdi_dict and country_code in hdi_dict[publication_year]:
            hdi_values.append(hdi_dict[publication_year][country_code])
        else:
            hdi_values.append(None)
    return hdi_values


# Path to the directory containing the TSV files
directory = "../data"

# List all TSV files starting with 'cs_works_'
tsv_files = [file for file in os.listdir(directory) if file.startswith("cs_works_")]

# Read all TSV files into a list of DataFrames
dfs = []
for file in tsv_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path, sep="\t")
    dfs.append(df)

# Merge all DataFrames into one
if dfs:
    merged_df = pd.concat(dfs)

# Remove duplicates
merged_df.drop_duplicates(inplace=True)
merged_df.dropna(subset=["countries", "institution_types"], inplace=True)

merged_df = merged_df[
    merged_df["institution_types"].apply(
        lambda x: all(
            inst_type in ["company", "education"] for inst_type in literal_eval(x)
        )
    )
]

updated_countries_list = []
other_columns = merged_df.drop(
    "countries", axis=1
)  # Extract all columns except 'countries'

for index, row in tqdm(
    merged_df.iterrows(), total=len(merged_df), desc="Processing rows"
):
    updated_countries = []
    for country in literal_eval(row["countries"]):
        if country in CN_COUNTRIES:  # Add other CN countries as needed
            updated_countries.append("CN")
        elif country == "XK":
            updated_countries.append("RS")
        else:
            updated_countries.append(country)

    # Append the updated 'countries' list to the main list
    updated_countries_list.append(updated_countries)

# Create a DataFrame from the updated 'countries' list
updated_countries_df = pd.DataFrame({"countries": updated_countries_list})

# Concatenate the 'other_columns' DataFrame with the 'updated_countries_df'
merged_df = pd.concat(
    [other_columns.reset_index(drop=True), updated_countries_df], axis=1
)

merged_df.to_csv("../data/temporal_merged_data.tsv", sep="\t", index=False)

# Load HDI data
dev_df = pd.read_csv("../data/human_dev_standard.csv")
hdi_dict = {}

# Populate the hdi_dict
for row in dev_df.itertuples():
    year = row.Year
    code = row.Code
    hdi = row.Hdi
    if year not in hdi_dict:
        hdi_dict[year] = {}
    hdi_dict[year][code] = hdi

# Process each row efficiently


# Update country relations, institution types, and HDI values
merged_df["country_relation"] = merged_df.apply(
    lambda row: process_row(row, EU_COUNTRIES, CN_COUNTRIES), axis=1
)
merged_df["institution_type"] = merged_df["institution_types"].apply(
    get_institution_type
)
merged_df["hdi"] = merged_df.apply(lambda row: get_hdi(row, hdi_dict), axis=1)

# Save the filtered and modified DataFrame to a single TSV file
merged_file_path = "../data/merged_data.tsv"
merged_df.to_csv(merged_file_path, sep="\t", index=False)

print(
    f"Filtered and merged data with country relation, institution type, and HDI values saved to {merged_file_path}"
)
