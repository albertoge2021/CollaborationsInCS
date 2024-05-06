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


def hdi_analysis(df):
    """mixed_df = df[df["country_relation"] == "Mixed"]
    rows=[]
    columns = mixed_df.columns  # Get the column names from the original DataFrame
    for row in tqdm(
        mixed_df.itertuples(),
        total=len(mixed_df),
        desc="Removing null rows",
    ):
        hdi_values = literal_eval(row.hdi)
        if any(value is None for value in hdi_values):
            continue
        rows.append(row._asdict())
    mixed_df = pd.DataFrame(rows, columns=columns, index=None)  # Pass column names when creating DataFrame
    describe_stats = mixed_df['cited_by_count'].describe()

    # Save descriptive statistics to a text file
    describe_stats.to_csv('results/hdi/cited_by_count_describe.txt', header=True)
    # Iterate through the rows and tag non-developed based on 'hdi' column
    non_developed_tag = []
    for row in tqdm(
        mixed_df.itertuples(),
        total=len(mixed_df),
        desc="Counting Countries for Mixed Relation",
    ):
        hdi_values = literal_eval(row.hdi)
        if any(value <= 0.549 for value in hdi_values):
            non_developed_tag.append(True)
        else:
            non_developed_tag.append(False)

    # Add 'non_developed' column to the DataFrame
    mixed_df["non_developed"] = non_developed_tag

    # Separate the DataFrame into two groups based on 'non_developed' column
    non_developed_group = mixed_df[mixed_df["non_developed"]]
    developed_group = mixed_df[~mixed_df["non_developed"]]

    # Perform statistical tests (e.g., t-test) to check for differences between the groups
    statistical_test_result = stats.ttest_ind(
        non_developed_group["cited_by_count"],
        developed_group["cited_by_count"],
        equal_var=False,  # Assuming unequal variances
    )

    # Print the statistical test result
    result_string = f"Statistical Test Result:\nT-statistic: {statistical_test_result.statistic}\nP-value: {statistical_test_result.pvalue}"

    # Save the result to a text file
    with open("results/hdi/non_developed_country_participating.txt", "w") as f:
        f.write(result_string)

    plt.figure(figsize=(10, 6))
    # Plot the average cited_by_count for each group
    plt.bar(
        ["Has Low-developed", "No Low-developed"],
        [
            non_developed_group["cited_by_count"].mean(),
            developed_group["cited_by_count"].mean(),
        ],
    )
    plt.xlabel("Development Status", fontsize=15)
    plt.ylabel("Average Citations", fontsize=15)
    plt.title(
        "Average Citations for Collaborations with at least a Low Developed Country",
        fontsize=17,
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("results/hdi/bar_plot_average_citations_low_developed.png")
    plt.close()

    # Iterate through the rows and tag high developed based on 'hdi' column
    high_developed_tag = []
    for row in tqdm(
        mixed_df.itertuples(),
        total=len(mixed_df),
        desc="Counting Countries for Mixed Relation",
    ):
        hdi_values = literal_eval(row.hdi)
        if any(value >= 0.799 for value in hdi_values):
            high_developed_tag.append(True)
        else:
            high_developed_tag.append(False)

    # Add 'high_developed' column to the DataFrame
    mixed_df["high_developed"] = high_developed_tag

    # Separate the DataFrame into two groups based on 'high_developed' column
    high_developed_group = mixed_df[mixed_df["high_developed"]]
    other_group = mixed_df[~mixed_df["high_developed"]]

    # Perform statistical tests (e.g., t-test) to check for differences between the groups
    statistical_test_result = stats.ttest_ind(
        high_developed_group["cited_by_count"],
        other_group["cited_by_count"],
        equal_var=False,  # Assuming unequal variances
    )

    result_string = f"Statistical Test Result:\nT-statistic: {statistical_test_result.statistic}\nP-value: {statistical_test_result.pvalue}"

    # Save the result to a text file
    with open("results/hdi/developed_country_participating.txt", "w") as f:
        f.write(result_string)
    # Plot the average cited_by_count for each group
    plt.figure(figsize=(10, 6))
    plt.bar(
        ["Has Very High Developed", "No Very High Developed"],
        [
            high_developed_group["cited_by_count"].mean(),
            other_group["cited_by_count"].mean(),
        ],
    )
    plt.xlabel("Development Status", fontsize=15)
    plt.ylabel("Average Citations", fontsize=15)
    plt.title(
        "Average Citations for Collaborations with at least a Very High Developed Country",
        fontsize=17,
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("results/hdi/bar_plot_average_citations_high_developed.png")
    plt.close()

    def calculate_mean(lst):
        if lst and all(isinstance(x, (int, float)) for x in lst):  # Check if list is not empty and contains only numerical values
            return np.mean(lst)
        else:
            return 0

    # Calculate Pearson correlation coefficient
    pearson_corr, pearson_p_value = pearsonr(
        mixed_df["cited_by_count"],
        mixed_df["hdi"].apply(lambda x: calculate_mean(literal_eval(x)))
    )

    # Calculate Spearman correlation coefficient
    spearman_corr, spearman_p_value = spearmanr(
        mixed_df["cited_by_count"],
        mixed_df["hdi"].apply(lambda x: calculate_mean(literal_eval(x)))
    )

    # Print and save the correlation coefficients
    correlation_result = f"Pearson Correlation Coefficient: {pearson_corr} (p-value: {pearson_p_value})\nSpearman Correlation Coefficient: {spearman_corr} (p-value: {spearman_p_value})"

    # Save the correlation result to a text file
    with open("results/hdi/correlation_result.txt", "w") as f:
        f.write(correlation_result)

    # Save the mean cited_by_count for each group to a text file (non-developed)
    mean_citations_non_developed = {
        "Has Non-developed": non_developed_group["cited_by_count"].mean(),
        "Has Only Developed": developed_group["cited_by_count"].mean(),
    }

    with open("results/hdi/mean_citations_non_developed.txt", "w") as f:
        f.write(
            "Mean Citations for Collaborations with at least a Low Developed Country:\n"
        )
        for group, mean in mean_citations_non_developed.items():
            f.write(f"{group}: {mean}\n")

    # Save the mean cited_by_count for each group to a text file (high-developed)
    mean_citations_high_developed = {
        "Has Very High Developed": high_developed_group["cited_by_count"].mean(),
        "Without Very High Developed": other_group["cited_by_count"].mean(),
    }

    with open("results/hdi/mean_citations_high_developed.txt", "w") as f:
        f.write(
            "Mean Citations for Collaborations with at least a Very High Developed Country:\n"
        )
        for group, mean in mean_citations_high_developed.items():
            f.write(f"{group}: {mean}\n")

    # Perform statistical tests (t-test) for non-developed group
    t_statistic_non_developed, p_value_non_developed = stats.ttest_ind(
        non_developed_group["cited_by_count"],
        developed_group["cited_by_count"],
        equal_var=False,  # Assuming unequal variances
    )

    # Save the t-test result to a text file (non-developed)
    result_string_non_developed = "APA Format Result for Collaborations with at least a Low Developed Country:\n\n"
    result_string_non_developed += f"T({len(non_developed_group['cited_by_count']) + len(developed_group['cited_by_count']) - 2:.0f}) = {t_statistic_non_developed:.2f}, p = {p_value_non_developed:.3f}\n\n"

    # Interpretation
    if p_value_non_developed < 0.05:
        result_string_non_developed += "The difference in cited_by_count between groups is statistically significant.\n"
    else:
        result_string_non_developed += "There is no statistically significant difference in cited_by_count between groups.\n"

    # Save the result to a text file
    with open("results/hdi/t_test_non_developed_apa.txt", "w") as f:
        f.write(result_string_non_developed)

    # Perform statistical tests (t-test) for high-developed group
    t_statistic_high_developed, p_value_high_developed = stats.ttest_ind(
        high_developed_group["cited_by_count"],
        other_group["cited_by_count"],
        equal_var=False,  # Assuming unequal variances
    )

    # Save the t-test result to a text file (high-developed)
    result_string_high_developed = "APA Format Result for Collaborations with at least a Very High Developed Country:\n\n"
    result_string_high_developed += f"T({len(high_developed_group['cited_by_count']) + len(other_group['cited_by_count']) - 2:.0f}) = {t_statistic_high_developed:.2f}, p = {p_value_high_developed:.3f}\n\n"

    # Interpretation
    if p_value_high_developed < 0.05:
        result_string_high_developed += "The difference in cited_by_count between groups is statistically significant.\n"
    else:
        result_string_high_developed += "There is no statistically significant difference in cited_by_count between groups.\n"

    # Save the result to a text file
    with open("results/hdi/t_test_high_developed_apa.txt", "w") as f:
        f.write(result_string_high_developed)

    # Calculate for non_developed_group
    num_retracted_non_developed = non_developed_group["is_retracted"].sum()
    total_works_non_developed = len(non_developed_group)
    percentage_retracted_non_developed = (
        num_retracted_non_developed / total_works_non_developed
    ) * 100
    num_retracted_developed = developed_group["is_retracted"].sum()
    total_works_developed = len(developed_group)
    percentage_retracted_developed = (
        num_retracted_developed / total_works_developed
    ) * 100

    # Save the results/hdi to a text file for non_developed_group
    with open("results/hdi/retracted_stats_non_developed.txt", "w") as f:
        f.write(
            f"Number of Retracted Works for Non-Developed Group: {num_retracted_non_developed}\n"
        )
        f.write(
            f"Total Number of Works for Non-Developed Group: {total_works_non_developed}\n"
        )
        f.write(
            f"Percentage of Retracted Works for Non-Developed Group: {percentage_retracted_non_developed:.2f}%\n"
        )
        f.write(
            f"Number of Retracted Works for Developed Group: {num_retracted_developed}\n"
        )
        f.write(f"Total Number of Works for Developed Group: {total_works_developed}\n")
        f.write(
            f"Percentage of Retracted Works for Developed Group: {percentage_retracted_developed:.2f}%\n"
        )

    # Calculate for high_developed_group
    num_retracted_high_developed = high_developed_group["is_retracted"].sum()
    total_works_high_developed = len(high_developed_group)
    percentage_retracted_high_developed = (
        num_retracted_high_developed / total_works_high_developed
    ) * 100
    num_retracted_other = other_group["is_retracted"].sum()
    total_works_other = len(other_group)
    percentage_retracted_other = (num_retracted_other / total_works_other) * 100

    # Save the results/hdi to a text file for high_developed_group
    with open("results/hdi/retracted_stats_high_developed.txt", "w") as f:
        f.write(
            f"Number of Retracted Works for High-Developed Group: {num_retracted_high_developed}\n"
        )
        f.write(
            f"Total Number of Works for High-Developed Group: {total_works_high_developed}\n"
        )
        f.write(
            f"Percentage of Retracted Works for High-Developed Group: {percentage_retracted_high_developed:.2f}%\n"
        )
        f.write(f"Number of Retracted Works for Other Group: {num_retracted_other}\n")
        f.write(f"Total Number of Works for Other Group: {total_works_other}\n")
        f.write(
            f"Percentage of Retracted Works for Other Group: {percentage_retracted_other:.2f}%\n"
        )"""
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
    df_all_not_only = df_all_not_only[condition_mask]
    colors = {
        "Mixed": "deepskyblue",
        "US-EU-CN": "limegreen",
        "Other countries": "orangered",
    }
    # Step 1: Calculate the number of retracted articles, total number of rows, and percentage by relation
    retraction_stats = (
        df_all_not_only.groupby("country_relation")["is_retracted"]
        .agg(["sum", "count"])
        .reset_index()
    )
    retraction_stats["percentage"] = (
        retraction_stats["sum"] / retraction_stats["count"]
    ) * 100

    # Step 2: Save the results to a CSV file
    retraction_stats.to_csv("results/hdi/retraction_stats.csv", index=False)

    # Step 3: Plot the percentages in a bar plot by relation
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="country_relation", y="percentage", data=retraction_stats, palette=colors
    )
    plt.xlabel("Relation", fontsize=15)
    plt.ylabel("Percentage of Retracted Articles", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Percentage of Retracted Articles by Relation", fontsize=17)
    plt.savefig(f"results/hdi/barplot_retracted_publications_per_relation.png")
    plt.close()
