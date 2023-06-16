from ast import literal_eval
from collections import Counter
import pandas as pd
import warnings
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

# Setup Data
dev_df = pd.read_csv("human_dev_standard.csv")
df = pd.read_csv("cs_eu.csv")

## GENERAL ANALYSIS


# Descriptive statistics
Path("computer_science/general_analysis_eu/").mkdir(parents=True, exist_ok=True)
df[["citations"]].describe().to_csv(
    "computer_science/general_analysis_eu/describe_max_distance_by_type.csv"
)
