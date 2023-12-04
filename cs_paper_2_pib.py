from ast import literal_eval
from collections import Counter, defaultdict
from itertools import combinations
import json
from matplotlib import patches
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, f_oneway
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from tqdm import tqdm
import geopandas as gpd
from geopy.geocoders import Nominatim
from pycountry_convert import country_alpha2_to_country_name
import matplotlib.pyplot as plt
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

# Setup Data
df = pd.read_csv("gdp_hdi_hemisphere_dataset.csv")
df.columns = [
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
df["Year"] = df["Year"].astype(int)

# pib-numero publicaciones
# hacer estudios por grupos basados en hdi (publicaciones, media de citationes, media de distancia, media de distancia maxima)
# hacer estudios basados en PIB
# Specify the columns for which you want to create regression plots
columns_of_interest = [
    "Mean_GDP",
    "Median_GDP",
    "Max_GDP",
    "Min_GDP",
    "Mean_HDI",
    "Median_HDI",
    "Max_HDI",
    "Min_HDI",
]

df = df.replace([np.inf, -np.inf], np.nan).dropna()

"""for col in columns_of_interest:
    # Create a regression plot
    sns.regplot(x=col, y='Citations', data=df)
    plt.title(f"Regression Plot for {col} and Citations")
    plt.xlabel(col)
    plt.ylabel('Citations')
    plt.savefig(f"paper_results_2/{col}_regression_plot.png")
    plt.close()"""


def test_1():
    # Perform Pearson and Spearman correlation tests and save the results in a text file
    correlation_results = []

    for col in columns_of_interest:
        pearson_corr, _ = pearsonr(df[col], df["Citations"])
        spearman_corr, _ = spearmanr(df[col], df["Citations"])

        correlation_results.append(
            {
                "Column": col,
                "Pearson Correlation": pearson_corr,
                "Spearman Correlation": spearman_corr,
            }
        )

    with open("paper_results_2/correlation_results.txt", "w") as file:
        for result in correlation_results:
            file.write(f"Column: {result['Column']}\n")
            file.write(f"Pearson Correlation: {result['Pearson Correlation']}\n")
            file.write(f"Spearman Correlation: {result['Spearman Correlation']}\n")

            # Interpretation
            if abs(result["Pearson Correlation"]) >= 0.7:
                file.write("Strong correlation.\n")
            elif 0.3 <= abs(result["Pearson Correlation"]) < 0.7:
                file.write("Moderate correlation.\n")
            else:
                file.write("Weak correlation.\n")

            if abs(result["Spearman Correlation"]) >= 0.7:
                file.write("Strong monotonic correlation.\n")
            elif 0.3 <= abs(result["Spearman Correlation"]) < 0.7:
                file.write("Moderate monotonic correlation.\n")
            else:
                file.write("Weak monotonic correlation.\n")

            file.write("\n")

    # Group by 'Hemisphere' column and count the number of entries in each group
    grouped = df.groupby("Hemisphere").count()

    # Calculate average and median Citations for each group
    avg_citations = df.groupby("Hemisphere")["Citations"].mean()
    median_citations = df.groupby("Hemisphere")["Citations"].median()

    # Perform Kruskal-Wallis test
    h_statistic, p_value = stats.kruskal(
        *[group["Citations"] for _, group in df.groupby("Hemisphere")]
    )

    # Open the file in append mode and write the results
    with open("paper_results_2/hemisphere_results.txt", "w") as file:
        file.write("Hemisphere Counts:\n")
        file.write(grouped.to_string() + "\n\n")

        file.write("Average Citations by Hemisphere:\n")
        file.write(avg_citations.to_string() + "\n\n")

        file.write("Median Citations by Hemisphere:\n")
        file.write(median_citations.to_string() + "\n\n")

        file.write(f"Kruskal Test between Hemispheres:\n")
        file.write(f"T-statistic: {h_statistic}\n")
        file.write(f"P-value: {p_value}\n\n")

        if p_value < 0.05:  # You can set your significance level here
            file.write(
                "Statistically significant differences exist between the groups.\n"
            )
        else:
            file.write(
                "No statistically significant differences exist between the groups.\n"
            )

    columns_of_interest = ["Mean_HDI", "Median_HDI", "Max_HDI", "Min_HDI"]
    # Define the bins for HDI
    bins = [0, 0.549, 0.699, 0.799, 1.0]
    labels = ["Low", "Medium", "High", "Very High"]

    with open("paper_results_2/grouped_hdi_results.txt", "w") as file:
        for column in columns_of_interest:
            # Create a new column 'HDI_Bin' based on the bins
            df["HDI_Bin"] = pd.cut(df[column], bins=bins, labels=labels)

            # Group by 'HDI_Bin' and calculate count, median, and mean of 'Citations'
            grouped_df = (
                df.groupby("HDI_Bin")["Citations"]
                .agg(["count", "median", "mean"])
                .reset_index()
            )

            # Perform Pearson and Spearman correlation tests
            pearson_corr, _ = pearsonr(df[column], df["Citations"])
            spearman_corr, _ = spearmanr(df[column], df["Citations"])

            # Save correlation test results in the text file
            file.write(
                f"Correlation tests for {column} and Citations in {df['HDI_Bin']} HDI Development:\n"
            )
            file.write(f"Pearson Correlation: {pearson_corr}\n")
            file.write(f"Spearman Correlation: {spearman_corr}\n")
            if abs(pearson_corr) >= 0.7 or abs(spearman_corr) >= 0.7:
                file.write(
                    "The strong correlation suggests a significant relationship between HDI and Citations.\n\n"
                )
            elif 0.3 <= abs(pearson_corr) < 0.7 or 0.3 <= abs(spearman_corr) < 0.7:
                file.write(
                    "The moderate correlation suggests a moderate relationship between HDI and Citations.\n\n"
                )
            else:
                file.write(
                    "The weak correlation suggests a weak or no relationship between HDI and Citations.\n\n"
                )

            file.write(f"Statistics for {column}:\n")
            file.write(f"HDI_Bin\tCount\tMedian\tMean\n")
            for index, row in grouped_df.iterrows():
                file.write(
                    f"{row['HDI_Bin']}\t{row['count']}\t{row['median']}\t{row['mean']}\n"
                )

                # Plot regression plot for each 'HDI_Bin'
                subset_df = df[df["HDI_Bin"] == row["HDI_Bin"]]
                sns.regplot(
                    x=column,
                    y="Citations",
                    data=subset_df,
                    label=row["HDI_Bin"],
                    scatter=False,
                )

                # Perform Pearson and Spearman correlation tests
                pearson_corr, _ = pearsonr(subset_df[column], subset_df["Citations"])
                spearman_corr, _ = spearmanr(subset_df[column], subset_df["Citations"])

                # Save correlation test results in the text file
                file.write(
                    f"Correlation tests for {column} and Citations in {row['HDI_Bin']} HDI Development:\n"
                )
                file.write(f"Pearson Correlation: {pearson_corr}\n")
                file.write(f"Spearman Correlation: {spearman_corr}\n")
                if abs(pearson_corr) >= 0.7 or abs(spearman_corr) >= 0.7:
                    file.write(
                        "The strong correlation suggests a significant relationship between HDI and Citations.\n\n"
                    )
                elif 0.3 <= abs(pearson_corr) < 0.7 or 0.3 <= abs(spearman_corr) < 0.7:
                    file.write(
                        "The moderate correlation suggests a moderate relationship between HDI and Citations.\n\n"
                    )
                else:
                    file.write(
                        "The weak correlation suggests a weak or no relationship between HDI and Citations.\n\n"
                    )

                plt.title(
                    f"Regression Plot for {column} and Citations in {row['HDI_Bin']} HDI Development"
                )
                plt.xlabel(column)
                plt.ylabel("Citations")
                plt.legend(title="HDI level")

                # Save the regression plot
                plt.savefig(
                    f"paper_results_2/{column}_regression_plot_{row['HDI_Bin']}.png"
                )
                plt.close()

            # Perform ANOVA test
            anova_result = f_oneway(
                *[df[df["HDI_Bin"] == label]["Citations"] for label in labels]
            )
            file.write(f"ANOVA Test for {column} and Citations among HDI levels:\n")
            file.write(f"F-statistic: {anova_result.statistic}\n")
            file.write(f"P-value: {anova_result.pvalue}\n")
            if anova_result.pvalue < 0.05:
                file.write(
                    "The ANOVA test indicates a statistically significant difference in Citations among HDI levels.\n\n"
                )
            else:
                file.write(
                    "The ANOVA test does not show a statistically significant difference in Citations among HDI levels.\n\n"
                )

            # Group by 'Year' and 'HDI_Bin' and calculate the count of rows
            grouped_df = (
                df.groupby(["Year", "HDI_Bin"]).size().reset_index(name="Count")
            )

            # Plot the number of rows by group per year using a bar plot
            sns.barplot(
                x="Year", y="Count", hue="HDI_Bin", data=grouped_df, palette="viridis"
            )
            plt.title(f"Number of Rows by {column} Group per Year")
            plt.xlabel("Year")
            plt.ylabel("Number of Rows")
            plt.legend(title="HDI level")

            # Save the bar plot
            plt.savefig(f"paper_results_2/bar_{column}_rows_by_group_per_year.png")
            plt.close()

            sns.lineplot(
                x="Year", y="Count", hue="HDI_Bin", data=grouped_df, palette="viridis"
            )
            plt.title(f"Number of Rows by {column} Group per Year")
            plt.xlabel("Year")
            plt.ylabel("Number of Rows")
            plt.legend(title="HDI level")

            # Save the bar plot
            plt.savefig(f"paper_results_2/line_{column}_rows_by_group_per_year.png")
            plt.close()


def test_2():
    min_gdp = df["Mean_GDP"].min()
    max_gdp = df["Mean_GDP"].max()

    # Define the bin edges based on min and max GDP
    bin_edges = [
        min_gdp,
        (min_gdp + max_gdp) / 4,
        2 * (min_gdp + max_gdp) / 4,
        3 * (min_gdp + max_gdp) / 4,
        max_gdp,
    ]

    # Define labels for the bins
    bin_labels = ["Low GDP", "Medium-Low GDP", "Medium-High GDP", "High GDP"]

    # Create a new column 'GDP_Group' based on the mean GDP bins
    df["GDP_Group"] = pd.cut(
        df["Mean_GDP"], bins=bin_edges, labels=bin_labels, include_lowest=True
    )

    # Group the DataFrame by the 'GDP_Group' column
    grouped_df = df.groupby("GDP_Group")

    with open("paper_results_2/gdp_statistics.txt", "w") as file:
        for group_name, group_data in grouped_df:
            min_threshold = bin_edges[bin_labels.index(group_name)]
            max_threshold = (
                bin_edges[bin_labels.index(group_name) + 1]
                if group_name != "High GDP"
                else max_gdp
            )

            file.write(f"Group: {group_name}\n")
            file.write(f"GDP Thresholds: {min_threshold} - {max_threshold}\n")
            file.write(f"Number of Rows: {len(group_data)}\n")
            file.write(f"Mean Citations: {group_data['Citations'].mean()}\n")
            file.write(f"Median Citations: {group_data['Citations'].median()}\n")
            file.write("\n")


def main():
    # test_1()
    test_2()


if __name__ == "__main__":
    main()
