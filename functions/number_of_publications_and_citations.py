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
from scipy.stats import f_oneway


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
EU_COUNTRIES_ISO3 = [
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
]


def number_of_publications_and_citations(df: pd.DataFrame, colors):
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
    ]
    """
    Function that returns the number of publications and citations.
    """
    # region basic analysis
    # Count the rows
    num_rows = df.shape[0]
    grouped_counts = df.groupby("country_relation").size()

    # Calculate percentages
    percentages = (grouped_counts / num_rows) * 100

    # Save the count to a text file
    with open(
        "results/num_papers_and_citations/general_esults.txt", "w"
    ) as output_file:
        output_file.write("Number of rows: " + str(num_rows) + "\n\n")
        output_file.write(
            "Number of rows by Country Relation: \n"
            + grouped_counts.to_string()
            + "\n\n"
        )
        output_file.write(
            "Percentage of each group:\n" + percentages.to_string() + "\n\n"
        )

    df_mixed_subset = df[df["country_relation"] == "Mixed"]
    df_other_countries = df[df["country_relation"] == "Other countries"]
    df_cn_eu_us = df[
        df["country_relation"].isin(["CN-EU-US", "EU-US", "CN-US", "CN-EU"])
    ]
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

    # Grouping by 'country_relation' and 'institution_type' and describing 'cited_by_count'
    description = df_all_not_only.groupby(["country_relation", "institution_type"])[
        "cited_by_count"
    ].describe()

    # Save the descriptive statistics to a CSV file
    description.to_csv(
        "results/num_papers_and_citations/descriptive_stats_not_only.csv"
    )

    # Create an empty dictionary to store the results
    country_summary = defaultdict(
        lambda: {
            "count_mixed": 0,
            "count_other": 0,
            "cited_by_mixed": 0,
            "cited_by_other": 0,
        }
    )

    # Iterate over each row in the DataFrame
    for row in tqdm(
        df_all_not_only.itertuples(), total=len(df_all_not_only), desc="Creating table"
    ):
        countries = literal_eval(row.countries)
        country_relation = row.country_relation
        cited_by_count = row.cited_by_count

        # Iterate over each country in the 'countries' list
        for country in countries:
            if country in EU_COUNTRIES or country in CN_COUNTRIES or country == "US":
                continue
            # Update counts and cited_by based on country_relation
            if country_relation == "Mixed":
                country_summary[country]["count_mixed"] += 1
                country_summary[country]["cited_by_mixed"] += cited_by_count
            elif country_relation == "Other countries":
                country_summary[country]["count_other"] += 1
                country_summary[country]["cited_by_other"] += cited_by_count

    # Calculate average cited_by_count for each country
    for country, summary in country_summary.items():
        count_mixed = summary["count_mixed"]
        count_other = summary["count_other"]
        cited_by_mixed = summary["cited_by_mixed"]
        cited_by_other = summary["cited_by_other"]

        if count_mixed > 0:
            country_summary[country]["avg_mixed"] = cited_by_mixed / count_mixed
        else:
            country_summary[country]["avg_mixed"] = 0

        if count_other > 0:
            country_summary[country]["avg_other"] = cited_by_other / count_other
        else:
            country_summary[country]["avg_other"] = 0

    # Write the results to a single CSV file
    with open("results/num_papers_and_citations/country_summary.csv", "w") as f:
        f.write(
            "country,count_mixed,count_other,avg_cited_by_mixed,avg_cited_by_other\n"
        )
        for country, summary in country_summary.items():
            count_mixed = summary["count_mixed"]
            count_other = summary["count_other"]
            avg_cited_by_mixed = summary["avg_mixed"]
            avg_cited_by_other = summary["avg_other"]
            f.write(
                f"{country},{count_mixed},{count_other},{avg_cited_by_mixed},{avg_cited_by_other}\n"
            )

    # Define the groups
    groups = ["country_relation"]

    # Calculate the desired statistics for each group
    grouped_stats = df.groupby(groups).agg(
        num_rows=("cited_by_count", "size"),  # Number of rows
        mean_citations=("cited_by_count", "mean"),  # Mean citations
        median_citations=("cited_by_count", "median"),  # Median citations
        num_retracted=(
            "is_retracted",
            lambda x: sum(x),
        ),  # Number of rows where is_retracted is True
    )

    # Bar plot for mean citations
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="country_relation",
        y="cited_by_count",
        data=df_all_not_only,
        estimator="mean",
        ci=None,
    )
    plt.title("Mean Citations by Relation", fontsize=17)
    plt.xlabel("Relation", fontsize=15)
    plt.ylabel("Mean Citations", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(
        f"results/num_papers_and_citations/barplot_mean_citations_per_relation.png"
    )
    plt.close()

    result_by_collaborators = {
        "collaborators": [],
        "retracted_count": [],
        "non_retracted_count": [],
        "percentage_retracted": [],
    }
    collaborators_citations = defaultdict(list)
    for row in tqdm(
        df_all_not_only.itertuples(),
        total=len(df_all_not_only),
        desc="Counting Collaborators",
    ):
        countries = literal_eval(row.countries)
        is_retracted = row.is_retracted
        citations = row.cited_by_count

        # Count the number of collaborators
        num_collaborators = len(countries)

        # Update result_by_collaborators dictionary
        if num_collaborators not in result_by_collaborators["collaborators"]:
            result_by_collaborators["collaborators"].append(num_collaborators)
            result_by_collaborators["retracted_count"].append(0)
            result_by_collaborators["non_retracted_count"].append(0)

        index = result_by_collaborators["collaborators"].index(num_collaborators)
        if is_retracted:
            result_by_collaborators["retracted_count"][index] += 1
        else:
            result_by_collaborators["non_retracted_count"][index] += 1
        collaborators_citations[num_collaborators].append(citations)

    # Calculate mean citations by the number of collaborators
    mean_citations_by_collaborators = {}
    for num_collaborators, citations_list in collaborators_citations.items():
        mean_citations_by_collaborators[num_collaborators] = sum(citations_list) / len(
            citations_list
        )

    # Filter data for collaborators up to 20
    filter_limit = 20
    filtered_data = {
        "collaborators": [],
        "percentage_retracted": [],
    }

    for i, collaborators in enumerate(result_by_collaborators["collaborators"]):
        filtered_data["collaborators"].append(collaborators)
        retracted_count = result_by_collaborators["retracted_count"][i]
        non_retracted_count = result_by_collaborators["non_retracted_count"][i]
        percentage_retracted = (
            (retracted_count / (retracted_count + non_retracted_count)) * 100
            if (retracted_count + non_retracted_count) > 0
            else 0
        )
        filtered_data["percentage_retracted"].append(percentage_retracted)

    # Convert data to a DataFrame
    temp_df = pd.DataFrame(filtered_data)

    # Check for normality using Shapiro-Wilk test
    stat, shapiro_p_value = shapiro(temp_df["percentage_retracted"])

    # Perform Pearson or Spearman correlation test based on normality result
    if shapiro_p_value > 0.05:  # If p-value > 0.05, assume normal distribution
        correlation_method = "Pearson"
        correlation_test = pearsonr
    else:
        correlation_method = "Spearman"
        correlation_test = spearmanr

    # Perform correlation test
    corr, p_value = correlation_test(
        temp_df["collaborators"], temp_df["percentage_retracted"]
    )

    # Save results in APA format to a text file
    with open(
        "results/num_papers_and_citations/correlation_results_for_collaborators.txt",
        "w",
    ) as f:
        f.write(
            f"Shapiro-Wilk Test for Normality: W({len(temp_df)}) = {stat:.3f}, p = {shapiro_p_value:.3f}\n"
        )
        f.write(
            f"{correlation_method} Correlation: r({len(temp_df) - 2}) = {corr:.3f}, p = {p_value:.3f}\n"
        )

    for relation in df_all_not_only["country_relation"].unique():
        # Filter the DataFrame by 'relation'
        relation_df = df_all_not_only[df_all_not_only["country_relation"] == relation]
        result_by_collaborators = {
            "collaborators": [],
            "retracted_count": [],
            "non_retracted_count": [],
            "percentage_retracted": [],
        }
        total_by_type = defaultdict(int)

        # Iterate through the DataFrame rows with tqdm
        for row in tqdm(
            relation_df.itertuples(),
            total=len(relation_df),
            desc="Counting Collaborators",
        ):
            countries = literal_eval(row.countries)
            is_retracted = row.is_retracted
            citations = row.cited_by_count
            # Count the number of collaborators
            num_collaborators = len(countries)

            # Update result_by_collaborators dictionary
            if num_collaborators not in result_by_collaborators["collaborators"]:
                result_by_collaborators["collaborators"].append(num_collaborators)
                result_by_collaborators["retracted_count"].append(0)
                result_by_collaborators["non_retracted_count"].append(0)

            index = result_by_collaborators["collaborators"].index(num_collaborators)
            if is_retracted:
                result_by_collaborators["retracted_count"][index] += 1
            else:
                result_by_collaborators["non_retracted_count"][index] += 1

        # Filter data for collaborators up to 20
        filter_limit = 20
        filtered_data = {
            "collaborators": [],
            "percentage_retracted": [],
        }

        for i, collaborators in enumerate(result_by_collaborators["collaborators"]):
            filtered_data["collaborators"].append(collaborators)
            retracted_count = result_by_collaborators["retracted_count"][i]
            non_retracted_count = result_by_collaborators["non_retracted_count"][i]
            percentage_retracted = (
                (retracted_count / (retracted_count + non_retracted_count)) * 100
                if (retracted_count + non_retracted_count) > 0
                else 0
            )
            filtered_data["percentage_retracted"].append(percentage_retracted)

        # Convert data to a DataFrame
        temp_df = pd.DataFrame(filtered_data)

        # Check for normality using Shapiro-Wilk test
        stat, shapiro_p_value = shapiro(temp_df["percentage_retracted"])

        # Perform Pearson or Spearman correlation test based on normality result
        if shapiro_p_value > 0.05:  # If p-value > 0.05, assume normal distribution
            correlation_method = "Pearson"
            correlation_test = pearsonr
        else:
            correlation_method = "Spearman"
            correlation_test = spearmanr

        # Perform correlation test
        corr, p_value = correlation_test(
            temp_df["collaborators"], temp_df["percentage_retracted"]
        )

        # Save results in APA format to a text file
        with open(
            f"results/num_papers_and_citations/correlation_results_for_collaborators_{relation}.txt",
            "w",
        ) as f:
            f.write(
                f"Shapiro-Wilk Test for Normality: W({len(temp_df)}) = {stat:.3f}, p = {shapiro_p_value:.3f}\n"
            )
            f.write(
                f"{correlation_method} Correlation: r({len(temp_df) - 2}) = {corr:.3f}, p = {p_value:.3f}\n"
            )

        # Filter DataFrame for relevant country_relation names
    country_relations = [
        "CN-EU",
        "CN-EU-US",
        "CN-US",
        "CN-only",
        "EU-US",
        "EU-only",
        "US-only",
    ]
    filtered_df = df[df["country_relation"].isin(country_relations)]

    # Perform ANOVA
    anova_result = f_oneway(
        filtered_df[filtered_df["country_relation"] == "CN-EU"]["cited_by_count"],
        filtered_df[filtered_df["country_relation"] == "CN-EU-US"]["cited_by_count"],
        filtered_df[filtered_df["country_relation"] == "CN-US"]["cited_by_count"],
        filtered_df[filtered_df["country_relation"] == "CN-only"]["cited_by_count"],
        filtered_df[filtered_df["country_relation"] == "EU-US"]["cited_by_count"],
        filtered_df[filtered_df["country_relation"] == "EU-only"]["cited_by_count"],
        filtered_df[filtered_df["country_relation"] == "US-only"]["cited_by_count"],
    )

    # Calculate degrees of freedom
    df_between = len(filtered_df["country_relation"].unique()) - 1
    df_within = len(filtered_df) - len(filtered_df["country_relation"].unique())

    # Print ANOVA result in APA format
    with open("results/num_papers_and_citations/anova_result.txt", "w") as f:
        f.write("ANOVA Result:\n")
        f.write(f"F-value: {anova_result.statistic}\n")
        f.write(f"p-value: {anova_result.pvalue}\n")
        f.write(f"Degrees of Freedom (between groups): {df_between}\n")
        f.write(f"Degrees of Freedom (within groups): {df_within}\n")
        f.write("\nAPA Format:\n")
        f.write(f"F({df_between}, {df_within}) = {anova_result.statistic}, p < .05\n")

    # Save the results to a CSV file
    grouped_stats.to_csv("results/num_papers_and_citations/grouped_statistics.csv")

    # Calculate the desired statistics for each group
    grouped_stats = df_all_not_only.groupby(groups).agg(
        num_rows=("cited_by_count", "size"),  # Number of rows
        mean_citations=("cited_by_count", "mean"),  # Mean citations
        median_citations=("cited_by_count", "median"),  # Median citations
        num_retracted=(
            "is_retracted",
            lambda x: sum(x),
        ),  # Number of rows where is_retracted is True
    )

    # Save the results to a CSV file
    grouped_stats.to_csv(
        "results/num_papers_and_citations/grouped_statistics_not_only.csv"
    )

    # Define the groups
    groups = ["country_relation", "institution_type"]

    # Calculate the desired statistics for each group
    grouped_stats = df.groupby(groups).agg(
        num_rows=("cited_by_count", "size"),  # Number of rows
        mean_citations=("cited_by_count", "mean"),  # Mean citations
        median_citations=("cited_by_count", "median"),  # Median citations
        num_retracted=(
            "is_retracted",
            lambda x: sum(x),
        ),  # Number of rows where is_retracted is True
    )

    # Save the results to a CSV file
    grouped_stats.to_csv(
        "results/num_papers_and_citations/grouped_statistics_by_country_and_institution.csv"
    )
    # Define the groups
    groups = ["institution_type"]

    # Calculate the desired statistics for each group
    grouped_stats = filtered_df.groupby(groups).agg(
        num_rows=("cited_by_count", "size"),  # Number of rows
        mean_citations=("cited_by_count", "mean"),  # Mean citations
        median_citations=("cited_by_count", "median"),  # Median citations
        num_retracted=(
            "is_retracted",
            lambda x: sum(x),
        ),  # Number of rows where is_retracted is True
    )

    # Save the results to a CSV file
    grouped_stats.to_csv(
        "results/num_papers_and_citations/grouped_statistics_by_institution.csv"
    )

    # Group by 'relation'
    grouped_by_relation = df_all_not_only.groupby("country_relation")

    # Define your groups for t-tests
    groups = df_all_not_only["country_relation"].unique()

    # Create a file to save the t-test results
    output_file = "results/num_papers_and_citations/t_test_results.txt"

    # Open the file in write mode
    with open(output_file, "w") as file:
        # Perform t-tests for each pair of groups
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1 = groups[i]
                group2 = groups[j]

                # Extract data for the two groups
                data_group1 = df_all_not_only[
                    df_all_not_only["country_relation"] == group1
                ]["cited_by_count"]
                data_group2 = df_all_not_only[
                    df_all_not_only["country_relation"] == group2
                ]["cited_by_count"]

                # Perform t-test
                t_statistic, p_value = ttest_ind(data_group1, data_group2)

                # Write results to the file
                file.write(f"T-test between {group1} and {group2}:\n")
                file.write(f"T-statistic: {t_statistic}\n")
                file.write(f"P-value: {p_value}\n")
                file.write("\n")

    # Save a txt file with the number of rows, number of is_retracted=True,
    # and the percentage of is_retracted=True / total rows for each group
    # Perform a statistical test to check for differences in average citations between groups
    grouped_citations = [group["cited_by_count"] for name, group in grouped_by_relation]

    # Perform ANOVA
    anova_result = stats.f_oneway(*grouped_citations)

    # Extract relevant information
    degrees_of_freedom_between = len(grouped_citations) - 1
    degrees_of_freedom_within = (
        len(grouped_citations[0]) - degrees_of_freedom_between - 1
    )
    f_statistic = anova_result.statistic
    p_value = anova_result.pvalue

    # Print the formatted result
    result_string = f"F({degrees_of_freedom_between}, {degrees_of_freedom_within}) = {f_statistic:.2f}, p = {p_value:.3f}"

    with open("results/num_papers_and_citations/grouped_stats.txt", "w") as f:
        # Print the ANOVA result
        f.write("ANOVA Result for citations:")
        f.write(str(result_string) + "\n")
        for name, group in grouped_by_relation:
            num_rows = len(group)
            num_retracted = group["is_retracted"].sum()
            percentage_retracted = (
                (num_retracted / num_rows) * 100 if num_rows > 0 else 0
            )

            f.write(f"Group: {name}\n")
            f.write(f"Number of Rows: {num_rows}\n")
            f.write(f"Number of is_retracted=True: {num_retracted}\n")
            f.write(f"Percentage of is_retracted=True: {percentage_retracted:.2f}%\n\n")
    # Initialize empty dictionaries to store results for each institution type

    unique_collaboration_types = df["institution_type"].unique()
    # Initialize empty dictionaries to store results for each institution type
    correlation_results_by_type = {}

    # Iterate over unique institution types
    for institution_type in unique_collaboration_types:
        us_ratio_total = []
        eu_ratio_total = []
        cn_ratio_total = []

        # Initialize citation counts for each institution type
        us_citations = 0
        eu_citations = 0
        cn_citations = 0

        # Iterate over filtered_df
        for row in tqdm(
            filtered_df[
                filtered_df["institution_type"] == institution_type
            ].itertuples()
        ):
            us_counts = 0
            eu_counts = 0
            cn_counts = 0
            country_list = literal_eval(row.countries)
            country_list = [
                "EU" if country in EU_COUNTRIES else country for country in country_list
            ]
            country_list = [
                "CN" if country in CN_COUNTRIES else country for country in country_list
            ]
            num_countries = row.num_participants
            cited_by_count = row.cited_by_count
            if "US" in country_list:
                us_counts += country_list.count("US")
                us_citations += cited_by_count
            if "CN" in country_list:
                cn_counts += country_list.count("CN")
                cn_citations += cited_by_count
            if "EU" in country_list:
                eu_counts += country_list.count("EU")
                eu_citations += cited_by_count
            if us_counts > 0:
                us_ratio_total.append(
                    ((us_counts / num_countries) * 100, cited_by_count)
                )
            if eu_counts > 0:
                eu_ratio_total.append(
                    ((eu_counts / num_countries) * 100, cited_by_count)
                )
            if cn_counts > 0:
                cn_ratio_total.append(
                    ((cn_counts / num_countries) * 100, cited_by_count)
                )

        # Calculate Spearman correlation for US, EU, and CN
        us_corr = spearmanr(
            [x[0] for x in us_ratio_total], [x[1] for x in us_ratio_total]
        )
        eu_corr = spearmanr(
            [x[0] for x in eu_ratio_total], [x[1] for x in eu_ratio_total]
        )
        cn_corr = spearmanr(
            [x[0] for x in cn_ratio_total], [x[1] for x in cn_ratio_total]
        )

        # Create a DataFrame for correlation results
        correlation_results = pd.DataFrame(
            {
                "Type": ["US", "EU", "CN"],
                "Correlation Coefficient": [
                    us_corr.correlation,
                    eu_corr.correlation,
                    cn_corr.correlation,
                ],
                "P-value": [us_corr.pvalue, eu_corr.pvalue, cn_corr.pvalue],
            }
        )

        # Calculate Spearman correlation in general
        general_corr = spearmanr(
            [
                item[0]
                for sublist in [us_ratio_total, eu_ratio_total, cn_ratio_total]
                for item in sublist
            ],
            [
                item[1]
                for sublist in [us_ratio_total, eu_ratio_total, cn_ratio_total]
                for item in sublist
            ],
        )

        # Append general correlation to results DataFrame
        correlation_results = correlation_results.append(
            {
                "Type": "General",
                "Correlation Coefficient": general_corr.correlation,
                "P-value": general_corr.pvalue,
            },
            ignore_index=True,
        )

        # Save results to a text file
        file_path = f"results/num_papers_and_citations/correlation_results_{institution_type}.txt"
        correlation_results.to_csv(file_path, sep="\t", index=False)

        # Store results in dictionary
        correlation_results_by_type[institution_type] = correlation_results

        # Calculate Spearman correlation for US, EU, and CN
        us_corr = pearsonr(
            [x[0] for x in us_ratio_total], [x[1] for x in us_ratio_total]
        )
        eu_corr = pearsonr(
            [x[0] for x in eu_ratio_total], [x[1] for x in eu_ratio_total]
        )
        cn_corr = pearsonr(
            [x[0] for x in cn_ratio_total], [x[1] for x in cn_ratio_total]
        )

        # Create a DataFrame for correlation results
        correlation_results = pd.DataFrame(
            {
                "Type": ["US", "EU", "CN"],
                "Correlation Coefficient": [
                    us_corr.correlation,
                    eu_corr.correlation,
                    cn_corr.correlation,
                ],
                "P-value": [us_corr.pvalue, eu_corr.pvalue, cn_corr.pvalue],
            }
        )

        # Calculate Spearman correlation in general
        general_corr = pearsonr(
            [
                item[0]
                for sublist in [us_ratio_total, eu_ratio_total, cn_ratio_total]
                for item in sublist
            ],
            [
                item[1]
                for sublist in [us_ratio_total, eu_ratio_total, cn_ratio_total]
                for item in sublist
            ],
        )

        # Append general correlation to results DataFrame
        correlation_results = correlation_results.append(
            {
                "Type": "General",
                "Correlation Coefficient": general_corr.correlation,
                "P-value": general_corr.pvalue,
            },
            ignore_index=True,
        )

        # Save results to a text file
        file_path = f"results/num_papers_and_citations/correlation_results_{institution_type}_pearson.txt"
        correlation_results.to_csv(file_path, sep="\t", index=False)

        # Store results in dictionary
        correlation_results_by_type[institution_type] = correlation_results

    # endregion

    # region maps

    collaborators = []

    for row in tqdm(df_all_not_only.itertuples()):
        country_list = set(literal_eval(row.countries))
        if len(set(country_list)) < 1:
            continue
        if len(country_list) < 1:
            continue
        for country in country_list:
            if country is None:
                continue
            country = "RS" if country == "XK" else country
            country = "CN" if country in CN_COUNTRIES else country
            country = pycountry.countries.get(alpha_2=country).alpha_3
            collaborators.append((row.country_relation, country))

    # participation_df = pd.DataFrame(collaborators, columns=["origin", "collaborator", "citations", "type"])
    participation_df = pd.DataFrame(collaborators, columns=["origin", "collaborator"])
    participation_df = participation_df[
        ~participation_df["collaborator"].isin(collaborators_to_remove)
    ]
    world_map = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    grouped_df = (
        participation_df.groupby(["origin", "collaborator"])
        .size()
        .reset_index(name="frequency")
    )

    for origin in participation_df["origin"].unique():
        subset_df = grouped_df[grouped_df["origin"] == origin]
        subset_df.to_csv(
            f"results/num_papers_and_citations/collaborators_{origin}.csv", index=False
        )
        merged_df = world_map.merge(
            subset_df, left_on="iso_a3", right_on="collaborator", how="left"
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        merged_df.plot(
            column="frequency",
            cmap="YlOrRd",
            linewidth=0.8,
            ax=ax,
            edgecolor="0.8",
            legend=True,
        )
        merged_df[merged_df["iso_a3"].isin(collaborators_to_remove)].plot(
            color="mediumpurple", ax=ax
        )
        if origin == "EU-only":
            origin_name = "with the European Union"
        elif origin == "US-only":
            origin_name = "with the United States of America"
        elif origin == "CN-only":
            origin_name = "with China"
        elif origin == "Other countries":
            origin_name = "with Other countries (without EU, US, and CN)"
        elif origin == "Mixed":
            origin_name = "with Mixed countries (Including EU, US, and CN)"
        elif origin == "US-EU-CN":
            origin_name = "between US-EU-CN"
        ax.set_title(f"Number of collaborations {origin_name}")
        plt.savefig(
            f"results/num_papers_and_citations/map_number_of_collaborations_{origin}.png"
        )
        plt.close()

    merged_df = world_map.merge(
        grouped_df, left_on="iso_a3", right_on="collaborator", how="left"
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    merged_df.plot(
        column="frequency",
        cmap="YlOrRd",
        linewidth=0.8,
        ax=ax,
        edgecolor="0.8",
        legend=True,
    )
    merged_df[merged_df["iso_a3"].isin(collaborators_to_remove)].plot(
        color="mediumpurple", ax=ax
    )
    ax.set_title(f"Number of collaborations")
    plt.savefig(f"results/num_papers_and_citations/map_number_of_collaborations.png")
    plt.close()

    # endregion

    # region publications by publication_year

    # Group by publication publication_year and count the number of rows for each group
    grouped_mixed = df_mixed_subset.groupby("publication_year").size()
    grouped_other_countries = df_other_countries.groupby("publication_year").size()
    df_cn_eu_us["country_relation"] = df_cn_eu_us["country_relation"].replace(
        {
            "CN-EU-US": "US-EU-CN",
            "EU-US": "US-EU-CN",
            "CN-US": "US-EU-CN",
            "CN-EU": "US-EU-CN",
        }
    )
    grouped_cn_eu_us = df_cn_eu_us.groupby("publication_year").size()

    # Plot the line plots
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_mixed.index, grouped_mixed.values, label="Mixed")
    plt.plot(
        grouped_other_countries.index,
        grouped_other_countries.values,
        label="Other countries",
    )
    plt.plot(grouped_cn_eu_us.index, grouped_cn_eu_us.values, label="US-EU-CN")
    plt.xlabel("Publication Year")
    plt.ylabel("Number of Rows")
    plt.title("Number of Rows by Publication Year")
    plt.legend()
    plt.grid(True)
    plt.close()

    # endregion
    # region insitutions

    df_cn_eu_us_onlies = df[
        df["country_relation"].isin(
            ["CN-EU-US", "EU-US", "CN-US", "CN-EU", "CN-only", "EU-only", "US-only"]
        )
    ]

    for collaboration_type in df_cn_eu_us_onlies["institution_type"].unique():
        collab_df = df_cn_eu_us_onlies[
            df_cn_eu_us_onlies["institution_type"] == collaboration_type
        ]
        us_collaborations = 0
        eu_collaborations = 0
        cn_collaborations = 0
        us_collaborations_total = 0
        eu_collaborations_total = 0
        cn_collaborations_total = 0
        us_eu_collaborations = 0
        us_cn_collaborations = 0
        eu_cn_collaborations = 0
        eu_cn_us_collaborations = 0
        us_citations = 0
        eu_citations = 0
        cn_citations = 0
        us_eu_citations = 0
        us_cn_citations = 0
        eu_cn_citations = 0
        eu_cn_us_citations = 0

        for row in tqdm(collab_df.itertuples()):
            country_list = literal_eval(row.countries)

            country_list = [
                "EU" if country in EU_COUNTRIES else country for country in country_list
            ]
            country_list = [
                "CN" if country in CN_COUNTRIES else country for country in country_list
            ]

            if "EU" in country_list or "CN" in country_list or "US" in country_list:
                check = True
            country_list = set(country_list)
            if check:
                citations = int(row.cited_by_count)
                if "US" in country_list:
                    us_collaborations_total += 1
                if "CN" in country_list:
                    cn_collaborations_total += 1
                if "EU" in country_list:
                    eu_collaborations_total += 1
                if (
                    "EU" in country_list
                    and "CN" in country_list
                    and "US" in country_list
                ):
                    eu_cn_us_collaborations += 1
                    eu_cn_us_citations += citations
                    continue
                elif "US" in country_list and "CN" in country_list:
                    us_cn_collaborations += 1
                    us_cn_citations += citations
                    continue
                elif "US" in country_list and "EU" in country_list:
                    us_eu_collaborations += 1
                    us_eu_citations += citations
                    continue
                elif "EU" in country_list and "CN" in country_list:
                    eu_cn_collaborations += 1
                    eu_cn_citations += citations
                    continue
                elif (
                    "US" in country_list
                    and "CN" not in country_list
                    and "EU" not in country_list
                    and len(country_list) == 1
                ):
                    us_collaborations += 1
                    us_citations += citations
                    continue
                elif (
                    "CN" in country_list
                    and "US" not in country_list
                    and "EU" not in country_list
                    and len(country_list) == 1
                ):
                    cn_collaborations += 1
                    cn_citations += citations
                    continue
                elif (
                    "EU" in country_list
                    and "US" not in country_list
                    and "CN" not in country_list
                    and len(country_list) == 1
                ):
                    eu_collaborations += 1
                    eu_citations += citations
                    continue

        with open(
            f"results/num_papers_and_citations/country_collaboration_cn_us_eu_percentage_type_{collaboration_type}.txt",
            "w",
        ) as file:
            file.write(
                f"US - US only collaboration represents {(us_collaborations / us_collaborations_total) * 100:.2f}% of total US collaborations\n"
            )
            file.write(
                f"CN - CN only collaboration represents {(cn_collaborations / cn_collaborations_total) * 100:.2f}% of total CN collaborations\n"
            )
            file.write(
                f"EU - EU only collaboration represents {(eu_collaborations / eu_collaborations_total) * 100:.2f}% of total EU collaborations\n"
            )
            file.write(
                f"CN - US collaboration represents {(us_cn_collaborations / us_collaborations_total) * 100:.2f}% of total US collaborations\n"
            )
            file.write(
                f"CN - US collaboration represents {(us_cn_collaborations / cn_collaborations_total) * 100:.2f}% of total Chinese collaborations\n"
            )
            file.write(
                f"CN - EU collaboration represents {(eu_cn_collaborations / cn_collaborations_total) * 100:.2f}% of total CN collaborations\n"
            )
            file.write(
                f"CN - EU collaboration represents {(eu_cn_collaborations / eu_collaborations_total) * 100:.2f}% of total EU collaborations\n"
            )
            file.write(
                f"EU - US collaboration represents {(us_eu_collaborations / us_collaborations_total) * 100:.2f}% of total US collaborations\n"
            )
            file.write(
                f"EU - US collaboration represents {(us_eu_collaborations / eu_collaborations_total) * 100:.2f}% of total EU collaborations\n"
            )
            file.write(
                f"EU - US - CN collaboration represents {(eu_cn_us_collaborations / us_collaborations_total) * 100:.2f}% of total US collaborations\n"
            )
            file.write(
                f"EU - US - CN collaboration represents {(eu_cn_us_collaborations / eu_collaborations_total) * 100:.2f}% of total EU collaborations\n"
            )
            file.write(
                f"EU - US - CN collaboration represents {(eu_cn_us_collaborations / cn_collaborations_total) * 100:.2f}% of total CN collaborations\n"
            )

        with open(
            f"results/num_papers_and_citations/country_collaboration_cn_us_eu_type_{collaboration_type}.txt",
            "w",
        ) as file:
            file.write(f"US - US only collaboration {(us_collaborations)}\n")
            file.write(f"CN - CN only collaboration {(cn_collaborations)}\n")
            file.write(f"EU - EU only collaboration {(eu_collaborations)}\n")
            file.write(f"CN - US collaboration {(us_cn_collaborations)}\n")
            file.write(f"CN - EU collaboration {(eu_cn_collaborations)}\n")
            file.write(f"EU - US collaboration {(us_eu_collaborations)}\n")
            file.write(f"EU - US - CN collaboration {(eu_cn_us_collaborations)}\n")

        # Define the data
        us_data = [us_collaborations, us_eu_collaborations, us_cn_collaborations]
        eu_data = [us_eu_collaborations, eu_collaborations, eu_cn_collaborations]
        cn_data = [us_cn_collaborations, eu_cn_collaborations, cn_collaborations]
        all_data = [
            eu_cn_us_collaborations,
            eu_cn_us_collaborations,
            eu_cn_us_collaborations,
        ]

        # Define the x-axis labels
        labels = ["US", "EU", "CN"]

        # Define the x-axis locations for each group of bars
        x_us = [0, 1, 2]
        x_eu = [5, 6, 7]
        x_cn = [10, 11, 12]
        x_all = [3, 8, 13]

        country_colors = ["deepskyblue", "limegreen", "orangered", "mediumpurple"]

        # Plot the bars
        plt.bar(x_us, us_data, color=country_colors, width=0.8, label="US")
        plt.bar(x_eu, eu_data, color=country_colors, width=0.8, label="EU")
        plt.bar(x_cn, cn_data, color=country_colors, width=0.8, label="CN")
        plt.bar(x_all, all_data, color=country_colors, width=0.8, label="EU-CN-US")

        # Add the x-axis labels and tick marks
        plt.xticks([1.5, 6.5, 11.5], labels)
        plt.xlabel("Country")
        plt.ylabel("Number of Collaborations")

        # Add a legend
        legend_colors = [patches.Patch(color=color) for color in country_colors]
        plt.legend(
            handles=legend_colors,
            labels=["US", "EU", "CN", "US-EU-CN"],
            title="Regions",
            loc="upper left",
        )

        # Show the plot
        if collaboration_type == "education":
            name = "Education-only"
        elif collaboration_type == "company":
            name = "Company-only"
        else:
            name = "Education and Company"
        plt.title(name + " publications")
        plt.savefig(
            f"results/num_papers_and_citations/bar_country_collaboration_cn_us_eu_type_{collaboration_type}.png"
        )
        plt.close()

        # Define the data
        us_data = [
            (us_collaborations / us_collaborations_total) * 100,
            (us_eu_collaborations / us_collaborations_total) * 100,
            (us_cn_collaborations / us_collaborations_total) * 100,
        ]
        eu_data = [
            (us_eu_collaborations / eu_collaborations_total) * 100,
            (eu_collaborations / eu_collaborations_total) * 100,
            (eu_cn_collaborations / eu_collaborations_total) * 100,
        ]
        cn_data = [
            (us_cn_collaborations / cn_collaborations_total) * 100,
            (eu_cn_collaborations / cn_collaborations_total) * 100,
            (cn_collaborations / cn_collaborations_total) * 100,
        ]
        all_data = [
            (eu_cn_us_collaborations / us_collaborations_total) * 100,
            (eu_cn_us_collaborations / eu_collaborations_total) * 100,
            (eu_cn_us_collaborations / cn_collaborations_total) * 100,
        ]

        # Define the x-axis labels
        labels = ["US", "EU", "CN"]

        # Define the x-axis locations for each group of bars
        x_us = [0, 1, 2]
        x_eu = [5, 6, 7]
        x_cn = [10, 11, 12]
        x_all = [3, 8, 13]

        # Plot the bars
        plt.bar(x_us, us_data, color=country_colors, width=0.8, label="US")
        plt.bar(x_eu, eu_data, color=country_colors, width=0.8, label="EU")
        plt.bar(x_cn, cn_data, color=country_colors, width=0.8, label="CN")
        plt.bar(x_all, all_data, color=country_colors, width=0.8, label="EU-CN-US")

        # Add the x-axis labels and tick marks
        plt.xticks([1.5, 6.5, 11.5], labels)
        plt.xlabel("Country")
        plt.ylabel("Percentage of Collaborations")

        # Add a legend
        legend_colors = [patches.Patch(color=color) for color in country_colors]
        plt.legend(
            handles=legend_colors,
            labels=["US", "EU", "CN", "US-EU-CN"],
            title="Regions",
            loc="best",
        )
        if collaboration_type == "education":
            name = "Education-only"
        elif collaboration_type == "company":
            name = "Company-only"
        else:
            name = "Education and Company"
        plt.title(name + " publications")
        plt.savefig(
            f"results/num_papers_and_citations/bar_collaboration_percent_type_{collaboration_type}.png"
        )
        plt.close()

        labels = ["US", "EU", "CN"]
        title = "Regions"
        color = "mediumpurple"

        plt.bar(labels, all_data, color=color)
        plt.title(title)
        plt.xlabel("Country")
        plt.ylabel("Percentage of Collaborations")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # Show the plot
        if collaboration_type == "education":
            name = "Education-only"
        elif collaboration_type == "company":
            name = "Company-only"
        else:
            name = "Education and Company"
        plt.title(name + " publications")
        plt.savefig(
            f"results/num_papers_and_citations/bar_collaboration_percent_type_{collaboration_type}_zoom.png"
        )
        plt.close()

    collaborations = []

    for row in tqdm(df_cn_eu_us_onlies.itertuples()):
        country_list = literal_eval(row.countries)
        country_list = [
            "EU" if country in EU_COUNTRIES else country for country in country_list
        ]
        country_list = [
            "CN" if country in CN_COUNTRIES else country for country in country_list
        ]
        check = False
        if "EU" in country_list or "CN" in country_list or "US" in country_list:
            check = True
        country_list = set(country_list)
        if check:
            citations = int(row.cited_by_count)
            if "EU" in country_list and "CN" in country_list and "US" in country_list:
                collaborations.append(
                    ("EU-CN-US", int(row.publication_year), citations)
                )
                continue
            elif "US" in country_list and "CN" in country_list:
                collaborations.append(("US-CN", int(row.publication_year), citations))
                continue
            elif "US" in country_list and "EU" in country_list:
                collaborations.append(("US-EU", int(row.publication_year), citations))
                continue
            elif "EU" in country_list and "CN" in country_list:
                collaborations.append(("EU-CN", int(row.publication_year), citations))
                continue
            elif (
                "US" in country_list
                and "CN" not in country_list
                and "EU" not in country_list
                and len(country_list) == 1
            ):
                collaborations.append(("US", int(row.publication_year), citations))
                continue
            elif (
                "CN" in country_list
                and "US" not in country_list
                and "EU" not in country_list
                and len(country_list) == 1
            ):
                collaborations.append(("CN", int(row.publication_year), citations))
                continue
            elif (
                "EU" in country_list
                and "US" not in country_list
                and "CN" not in country_list
                and len(country_list) == 1
            ):
                collaborations.append(("EU", int(row.publication_year), citations))
                continue

    df = pd.DataFrame(
        collaborations, columns=["relation", "publication_year", "citations"]
    )
    means = (
        df.groupby(["relation", "publication_year"]).size().reset_index(name="count")
    )
    sns.lineplot(data=means, x="publication_year", y="count", hue="relation")
    plt.xlabel("Year")
    plt.legend(title="Regions")
    plt.ylabel("Number of collaborations")
    plt.title("In-house and international collaborations")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration.png"
    )
    plt.close()

    means = means[means["publication_year"] >= 2010]
    sns.lineplot(data=means, x="publication_year", y="count", hue="relation")
    plt.xlabel("Year")
    plt.legend(title="Regions")
    plt.ylabel("Number of collaborations")
    plt.title("In-house and international collaborations")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration_10.png"
    )
    plt.close()

    means = (
        df.groupby(["relation", "publication_year"]).size().reset_index(name="count")
    )
    means = means[~means["relation"].isin(["CN", "US", "EU"])]
    sns.lineplot(data=means, x="publication_year", y="count", hue="relation")
    plt.xlabel("Year")
    plt.legend(title="Regions")
    plt.ylabel("Number of collaborations")
    plt.title("International collaborations only")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration_collab.png"
    )
    plt.close()

    means = means[means["publication_year"] >= 2010]
    sns.lineplot(data=means, x="publication_year", y="count", hue="relation")
    plt.xlabel("Year")
    plt.legend(title="Regions")
    plt.ylabel("Number of collaborations")
    plt.title("International collaborations only")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration_collab_10.png"
    )
    plt.close()

    means = (
        df_all_not_only.groupby(["country_relation", "publication_year"])
        .size()
        .reset_index(name="count")
    )
    sns.lineplot(data=means, x="publication_year", y="count", hue="country_relation")
    plt.xlabel("Year")
    plt.legend(title="Regions")
    plt.ylabel("Number of collaborations")
    plt.title("Number of collaborations by regions per year")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration_not_only.png"
    )
    plt.close()

    means = means[means["publication_year"] >= 2010]
    sns.lineplot(data=means, x="publication_year", y="count", hue="country_relation")
    plt.xlabel("Year")
    plt.legend(title="Regions")
    plt.ylabel("Number of collaborations")
    plt.title("Number of collaborations by regions per year")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration_10_not_only.png"
    )
    plt.close()

    means = (
        df_all_not_only.groupby(["country_relation", "publication_year"])
        .size()
        .reset_index(name="count")
    )
    means = means[~means["country_relation"].isin(["CN", "US", "EU"])]
    sns.lineplot(data=means, x="publication_year", y="count", hue="country_relation")
    plt.xlabel("Year")
    plt.legend(title="Regions")
    plt.ylabel("Number of collaborations")
    plt.title("Number of collaborations by regions per year")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration_collab_not_only.png"
    )
    plt.close()

    means = means[means["publication_year"] >= 2010]
    sns.lineplot(data=means, x="publication_year", y="count", hue="country_relation")
    plt.xlabel("Year")
    plt.legend(title="Regions")
    plt.ylabel("Number of collaborations")
    plt.title("Number of collaborations by regions per year")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration_collab_10_not_only.png"
    )
    plt.close()

    """    df = pd.DataFrame(
            collaborations, columns=["country_relation", "publication_year", "citations"]
        )
        means = (
            df.groupby(["relation", "publication_year"])["citations"]
            .mean()
            .reset_index(name="mean")
        )
        sns.lineplot(data=means, x="publication_year", y="mean", hue="country_relation")
        plt.xlabel("Year")
        plt.ylabel("Mean citations")
        plt.title("International collaborations only")
        plt.savefig(
            f"results/num_papers_and_citations/lineplot_mean_citations_per_year_per_collaboration.png"
        )
        plt.close()
    """
