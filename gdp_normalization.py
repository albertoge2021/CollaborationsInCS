from ast import literal_eval
from collections import Counter, defaultdict
from io import StringIO
from itertools import combinations
import json
from matplotlib import patches
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from tqdm import tqdm
import geopandas as gpd
from geopy.geocoders import Nominatim
from pycountry_convert import country_alpha3_to_country_alpha2
import matplotlib.pyplot as plt
from pycountry_convert import country_alpha2_to_country_name
import networkx as nx
import numpy as np
import squarify
import pandas as pd
import pycountry
import pycountry_convert as pc
from scipy.stats import chi2_contingency
import scipy.stats as stats
import seaborn as sns
import warnings


warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

"""# Setup Data
gdp_df = pd.read_csv("gdp_dataset.csv")
# Reshape the DataFrame using melt
df_melted = pd.melt(gdp_df, id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
                    var_name="Year", value_name="GDP")

# Function to convert ISO3 to ISO2
def convert_iso3_to_iso2(iso3_code):
    try:
        iso2_code = country_alpha3_to_country_alpha2(iso3_code)
        return iso2_code
    except Exception as e:
        print(f"Error converting {iso3_code} to ISO2: {e}")
        return None

# Apply the conversion function to the 'ISO3' column
df_melted['ISO2'] = df_melted['Country Code'].apply(lambda x: convert_iso3_to_iso2(x))

# Save the melted DataFrame to a new CSV file
df_melted.to_csv("gdp_dataset_normalized.csv", index=False)"""

dev_df = pd.read_csv("human_dev_standard.csv")
gdp_df = pd.read_csv("gdp_dataset_normalized.csv")
file_path = "data_countries/combined_data.tsv"
df = pd.read_csv(file_path, sep="\t")
eng_df = pd.read_csv("english_level.csv")
df = df[df["publication_year"] > 1989]
df = df[df["publication_year"] < 2022]
df = df.drop_duplicates()

gdp_dict = {}

for index, row in gdp_df.iterrows():
    iso2 = row["ISO2"]
    year = row["Year"]
    gdp = row["GDP"]

    if iso2 not in gdp_dict:
        gdp_dict[iso2] = {}

    gdp_dict[iso2][year] = gdp

hdi_dict = {}

# Populate the hdi_dict
for row in dev_df.itertuples():
    year = row.Year
    code = row.Code
    hdi = row.Hdi
    if year not in hdi_dict:
        hdi_dict[year] = {}
    hdi_dict[year][code] = hdi

gdp_data = []


def convert_country_code_to_name(country_code):
    try:
        country_name = country_alpha2_to_country_name(country_code)
        return country_name
    except LookupError:
        return None


def get_hemisphere(country_code):
    country_name = convert_country_code_to_name(country_code)

    if country_name:
        geolocator = Nominatim(user_agent="country_hemisphere_checker")

        # Get the capital city of the country
        location = geolocator.geocode(country_name, exactly_one=True)

        if location:
            latitude = location.latitude
            return latitude


import json

# Specify the path to your JSON file
file_path = 'country_latitudes.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    country_latitudes = json.load(file)

for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    country_set = literal_eval(row.countries)
    gdp_values = []
    hdi_values = []
    new_country_set = []
    for country in country_set:
        if country == "XK":
            country = "RS"
        if country == "TW":
            country = "CN"
        if country == "GP":
            country = "FR"
        new_country_set.append(country)
    for country in new_country_set:
        if country in gdp_dict:
            gdp_values.append(gdp_dict[country][str(row.publication_year)])
        if country in hdi_dict[year]:
            hdi_values.append(hdi_dict[year][country])

    checked_countries = {}
    northern = False
    southern = False
    for country in new_country_set:
        if country in checked_countries:
            continue
        else:
            checked_countries[country] = True
        # Check if the latitude is already stored
        if country not in country_latitudes:
            geolocator = Nominatim(user_agent="country_hemisphere_checker")
            country_latitudes[country] = geolocator.geocode(
                country
            ).latitude
        latitude = country_latitudes[country]
        if latitude > 0:
            northern = True
        elif latitude < 0:
            southern = True

    if northern and southern:
        hemisphere = "M"
    elif northern:
        hemisphere = "N"
    elif southern:
        hemisphere = "S"

    gdp_data.append(
        (
            row.title,
            row.publication_year,
            row.cited_by_count,
            row.type,
            row.is_retracted,
            row.institution_types,
            row.countries,
            row.concepts,
            gdp_values,
            hdi_values,
            hemisphere,
        )
    )

columns = [
    "title",
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
]

# Create a DataFrame using pib_data and columns
result_df = pd.DataFrame(gdp_data, columns=columns)
result_df.to_csv("data_countries/cs_works.csv", index=False)
