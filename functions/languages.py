from ast import literal_eval
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from scipy.stats import ttest_ind
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import shapiro

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


def language_analysis(df, colors):
    """
    Function that returns statistical analysis for the language of the works
    """
    # Grouping by 'country_relation' and 'institution_type' to get counts of languages

    df_all_not_only = df[
        df["country_relation"].isin(
            ["CN-EU-US", "EU-US", "CN-US", "CN-EU", "Mixed", "Other countries"]
        )
    ]
    df_all_not_only["country_relation"] = df_all_not_only["country_relation"].replace(
        {
            "CN-EU-US": "US-EU-CN",
            "EU-US": "US-EU-CN",
            "CN-US": "US-EU-CN",
            "CN-EU": "US-EU-CN",
        }
    )
    # Convert string representation of list to actual list and check the condition
    condition_mask = df_all_not_only["countries"].apply(
        lambda x: len(set(literal_eval(x))) > 1
    )

    # Filter the DataFrame based on the condition mask
    df = df_all_not_only[condition_mask]
    grouped = (
        df.groupby(["country_relation"])["language"]
        .value_counts()
        .reset_index(name="count")
    )

    # Saving grouped data to CSV files
    grouped.to_csv(
        "results/language/language_distribution_by_country_relation.csv", index=False
    )

    # Creating pie charts
    for index, group in grouped.groupby(["country_relation"]):
        country_relation = index
        plt.figure(figsize=(8, 8))
        plt.title(f"Language distribution for {country_relation}")

        # Calculate percentages
        total_count = group["count"].sum()
        group["percentage"] = (group["count"] / total_count) * 100

        # Determine whether each language percentage is greater than 1%
        group["language"] = group.apply(
            lambda row: row["language"] if row["percentage"] > 1 else "Other", axis=1
        )

        # Plot pie chart
        plt.pie(group["count"], labels=group["language"], autopct="%1.1f%%")
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

        plt.tight_layout()
        plt.savefig(f"results/language/language_distribution_{country_relation}.png")
        plt.close()

    # Grouping by 'country_relation' and 'institution_type' to get counts of languages
    grouped = (
        df.groupby(["country_relation", "institution_type"])["language"]
        .value_counts()
        .reset_index(name="count")
    )

    # Saving grouped data to CSV files
    grouped.to_csv(
        "results/language/language_distribution_by_country_relation_and_institution_type.csv",
        index=False,
    )

    # Creating pie charts
    for index, group in grouped.groupby(["country_relation", "institution_type"]):
        country_relation, institution_type = index
        plt.figure(figsize=(8, 8))
        plt.title(
            f"Language distribution for {country_relation} and {institution_type}"
        )

        # Calculate percentages
        total_count = group["count"].sum()
        group["percentage"] = (group["count"] / total_count) * 100

        # Determine whether each language percentage is greater than 1%
        group["language"] = group.apply(
            lambda row: row["language"] if row["percentage"] > 1 else "Other", axis=1
        )

        # Plot pie chart
        plt.pie(group["count"], labels=group["language"], autopct="%1.1f%%")
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

        plt.tight_layout()
        plt.savefig(
            f"results/language/language_distribution_{country_relation}_{institution_type}.png"
        )
        plt.close()

    language_counts = df["language"].value_counts().reset_index(name="count")
    language_counts.to_csv(
        "results/language/language_distribution_all_rows.csv", index=False
    )
