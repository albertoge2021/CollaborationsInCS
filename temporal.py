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

df = pd.read_csv("data_countries/cs_works.csv")

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
    for country in countries:
        if country in EU_COUNTRIES:
            country = "EU"
        if country in CN_COUNTRIES:
            country = "CN"
        country_list.append(country)

    # Check if all values are in ["US", "EU", "CN"]
    if all(country in ["US", "EU", "CN"] for country in country_list):
        relation = "US-EU-CN"
    elif any(country in ["US", "EU", "CN"] for country in country_list):
        relation = "Mixed"
    else:
        relation = "Others"

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

relation_groups = dict(tuple(new_df.groupby('relation')))

# Save each group into a separate file
for relation, df_group in relation_groups.items():
    filename = f"data_countries/{relation}_file.csv"
    df_group.to_csv(filename, index=False)
    print(f"Saved {relation} data to {filename}")