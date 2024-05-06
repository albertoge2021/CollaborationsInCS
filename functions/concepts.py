from ast import literal_eval
from collections import defaultdict
import csv
from matplotlib import patches
from scipy.stats import pearsonr, spearmanr
from scipy.stats import ttest_ind
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import shapiro
import pycountry


def concept_analysis(df):
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
        "Algorithm": "gold",
        "Telecommunications": "coral",
        "Philosophy": "indigo",
        "Political science": "azure",
        "Business": "lavender",
    }

    concepts_list = []

    for row in tqdm(
        df.itertuples(),
        total=len(df),
        desc="Counting Institution Types and Concepts",
    ):
        concepts = set(literal_eval(row.concepts))
        country_relation = str(row.country_relation)

        for concept in concepts:
            concepts_list.append((country_relation, concept, row.publication_year))

    # Convert dictionaries to DataFrames
    concepts_df = pd.DataFrame(
        concepts_list, columns=["country_relation", "concept", "year"]
    )

    for country_relation in concepts_df["country_relation"].unique():
        temp_df = concepts_df[concepts_df["country_relation"] == country_relation]
        temp_df = temp_df[temp_df["concept"] != "Computer science"]
        df_grouped = temp_df.groupby(["concept"]).size().reset_index(name="count")
        top_concepts_by_year = df_grouped.sort_values(by="count", ascending=False)
        top_concepts_by_year = top_concepts_by_year.head(10)
        sns.barplot(
            data=top_concepts_by_year, x="concept", y="count", palette=concept_colors
        )
        plt.xlabel("Concept", fontsize=15)
        plt.ylabel("Number of occurrences", fontsize=15)
        plt.title("10 Most reapeated concepts for " + country_relation, fontsize=17)
        plt.xticks(rotation=45, ha="right", fontsize=15)
        plt.savefig(
            f"results/concepts/{country_relation.lower().replace(' ', '_')}_top_concepts.png",
            bbox_inches="tight",
        )
        plt.close()

        df_grouped = (
            temp_df.groupby(["year", "concept"]).size().reset_index(name="count")
        )
        top_concepts_2021 = df_grouped[df_grouped["year"] == 2021].nlargest(10, "count")
        top_concepts_2021.to_csv(
            "results/concepts/top_concepts_2021.txt", sep="\t", index=False
        )
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
        plt.title(
            "10 Most reapeated concepts for " + country_relation + " by year",
            fontsize=17,
        )
        plt.xlabel("Year", fontsize=15)
        plt.ylabel("Number of occurrences", fontsize=15)
        plt.legend(title="Concept", loc="upper left")
        plt.savefig(
            f"results/concepts/only/{country_relation.lower().replace(' ', '_')}_top_concepts_by_year.png",
            bbox_inches="tight",
        )
        plt.close()

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

    concepts_list = []

    for row in tqdm(
        df_all_not_only.itertuples(),
        total=len(df_all_not_only),
        desc="Counting Institution Types and Concepts",
    ):
        concepts = set(literal_eval(row.concepts))
        country_relation = str(row.country_relation)

        for concept in concepts:
            concepts_list.append((country_relation, concept, row.publication_year))

    # Convert dictionaries to DataFrames
    concepts_df = pd.DataFrame(
        concepts_list, columns=["country_relation", "concept", "year"]
    )

    for country_relation in concepts_df["country_relation"].unique():
        temp_df = concepts_df[concepts_df["country_relation"] == country_relation]
        temp_df = temp_df[temp_df["concept"] != "Computer science"]
        df_grouped = temp_df.groupby(["concept"]).size().reset_index(name="count")
        top_concepts_by_year = df_grouped.sort_values(by="count", ascending=False)
        top_concepts_by_year = top_concepts_by_year.head(10)
        sns.barplot(
            data=top_concepts_by_year, x="concept", y="count", palette=concept_colors
        )
        plt.xlabel("Concept", fontsize=15)
        plt.ylabel("Number of occurrences", fontsize=15)
        plt.title("10 Most reapeated concepts for " + country_relation, fontsize=17)
        plt.xticks(rotation=45, ha="right", fontsize=15)
        plt.savefig(
            f"results/concepts/{country_relation.lower().replace(' ', '_')}_top_concepts.png",
            bbox_inches="tight",
        )
        plt.close()

        df_grouped = (
            temp_df.groupby(["year", "concept"]).size().reset_index(name="count")
        )
        top_concepts_2021 = df_grouped[df_grouped["year"] == 2021].nlargest(10, "count")
        top_concepts_2021.to_csv(
            "results/concepts/top_concepts_2021.txt", sep="\t", index=False
        )
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
        plt.title(
            "10 Most reapeated concepts for " + country_relation + " by year",
            fontsize=17,
        )
        plt.xlabel("Year", fontsize=15)
        plt.ylabel("Number of occurrences", fontsize=15)
        plt.legend(title="Concept", loc="upper left")
        plt.savefig(
            f"results/concepts/{country_relation.lower().replace(' ', '_')}_top_concepts_by_year.png",
            bbox_inches="tight",
        )
        plt.close()
