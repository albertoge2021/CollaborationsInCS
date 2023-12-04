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
df = pd.read_csv("cs_dataset_final.csv")

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


country_latitudes = {}

for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    country_set = literal_eval(row.countries)
    if len(set(country_set)) < 2:
        continue
    gdp_values = []
    hdi_values = []
    for country in country_set:
        if country == "XK":
            country = "RS"
        if country == "TW":
            country = "CN"
        if country == "GP":
            country = "FR"
        if country in gdp_dict:
            gdp_values.append(gdp_dict[country][str(row.year)])
        if country in hdi_dict[year]:
            hdi_values.append(hdi_dict[year][country])

    set_gdp_values = []
    set_hdi_values = []
    for country in set(country_set):
        if country == "XK":
            country = "RS"
        if country == "TW":
            country = "CN"
        if country == "GP":
            country = "FR"
        if country in gdp_dict:
            set_gdp_values.append(gdp_dict[country][str(row.year)])
        if country in hdi_dict[year]:
            set_hdi_values.append(hdi_dict[year][country])

    location_list = literal_eval(row.locations)
    checked_countries = {}
    northern = False
    southern = False
    for location in location_list:
        if location["country"] in checked_countries:
            continue
        else:
            checked_countries[location["country"]] = True
        if location["lat"] is None:
            # Check if the latitude is already stored
            if location["country"] not in country_latitudes:
                geolocator = Nominatim(user_agent="country_hemisphere_checker")
                country_latitudes[location["country"]] = geolocator.geocode(
                    location["country"]
                ).latitude
            latitude = country_latitudes[location["country"]]
        else:
            latitude = location["lat"]
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
            row.year,
            row.citations,
            row.type,
            literal_eval(row.countries),
            np.mean(set_gdp_values) if set_gdp_values else None,  # Mean GDP
            np.median(gdp_values) if gdp_values else None,  # Median GDP
            np.max(gdp_values) if gdp_values else None,  # Max GDP
            np.min(gdp_values) if gdp_values else None,  # Min GDP
            np.mean(set_hdi_values) if set_hdi_values else None,  # Mean HDI
            np.median(hdi_values) if hdi_values else None,  # Median HDI
            np.max(hdi_values) if hdi_values else None,  # Max HDI
            np.min(hdi_values) if hdi_values else None,  # Min HDI
            hemisphere,
        )
    )

columns = [
    "Year",
    "Citations",
    "Type",
    "Countries",
    "Mean_GDP",
    "Median_GDP",
    "Max_GDP",
    "Min_GDP",
    "Mean_HDI",
    "Median_HDI",
    "Max_HDI",
    "Min_HDI",
    "Hemisphere",
]

# Create a DataFrame using pib_data and columns
result_df = pd.DataFrame(gdp_data, columns=columns)
result_df.to_csv("gdp_hdi_hemisphere_dataset.csv", index=False)
