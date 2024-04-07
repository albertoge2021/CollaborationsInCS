from ast import literal_eval
from collections import Counter, defaultdict
import csv
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
from functions.hdi import hdi_analysis
from functions.languages import language_analysis

from functions.number_of_publications_and_citations import number_of_publications_and_citations


warnings.simplefilter(action="ignore", category=FutureWarning)

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

dataframe = pd.read_csv("data/merged_data.tsv", sep='\t')
dataframe = dataframe[(dataframe['publication_year'] >= 1990) & (dataframe['publication_year'] <= 2021)]

number_of_publications_and_citations(dataframe, colors)
#language_analysis(dataframe, colors)
#hdi_analysis(dataframe)