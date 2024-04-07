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
    condition_mask = df_all_not_only["countries"].apply(lambda x: len(set(literal_eval(x))) > 1)

    # Filter the DataFrame based on the condition mask
    df_all_not_only = df_all_not_only[condition_mask]

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

    # endregion

    # region statistical analysis

    collaborations = {
        "US": {},
        "EU": {},
        "CN": {},
        "US_EU": {},
        "US_CN": {},
        "CN_EU": {},
        "US_EU_CN": {},
        "OTHERS": {},
    }

    for row in tqdm(df.itertuples(), total=len(df), desc="Counting Collaborators"):
        country_list = set(literal_eval(row.countries))
        country_codes = []
        for country in country_list:
            country = "RS" if country == "XK" else country
            if country in EU_COUNTRIES:
                country_codes.append("EU")
            elif country in CN_COUNTRIES:
                country_codes.append("CN")
            else:
                country_codes.append(country)

        if (
            "US" in country_codes
            and "EU" not in country_codes
            and "CN" not in country_codes
        ):
            for code in country_codes:
                if code != "US":
                    collaborations["US"][code] = collaborations["US"].get(code, 0) + 1

        elif (
            "EU" in country_codes
            and "US" not in country_codes
            and "CN" not in country_codes
        ):
            for code in country_codes:
                if code != "EU":
                    collaborations["EU"][code] = collaborations["EU"].get(code, 0) + 1

        elif (
            "CN" in country_codes
            and "US" not in country_codes
            and "EU" not in country_codes
        ):
            for code in country_codes:
                if code != "CN":
                    collaborations["CN"][code] = collaborations["CN"].get(code, 0) + 1

        elif (
            "CN" in country_codes
            and "US" in country_codes
            and "EU" not in country_codes
        ):
            for code in country_codes:
                if code != "CN" and code != "US":
                    collaborations["US_CN"][code] = (
                        collaborations["US_CN"].get(code, 0) + 1
                    )

        elif (
            "CN" in country_codes
            and "US" not in country_codes
            and "EU" in country_codes
        ):
            for code in country_codes:
                if code != "CN" and code != "EU":
                    collaborations["CN_EU"][code] = (
                        collaborations["CN_EU"].get(code, 0) + 1
                    )

        elif (
            "CN" not in country_codes
            and "US" in country_codes
            and "EU" in country_codes
        ):
            for code in country_codes:
                if code != "US" and code != "EU":
                    collaborations["US_EU"][code] = (
                        collaborations["US_EU"].get(code, 0) + 1
                    )

        elif "US" in country_codes and "EU" in country_codes and "CN" in country_codes:
            for code in country_codes:
                if code != "US" and code != "EU" and code != "CN":
                    collaborations["US_EU_CN"][code] = (
                        collaborations["US_EU_CN"].get(code, 0) + 1
                    )

        elif (
            "US" not in country_codes
            and "EU" not in country_codes
            and "CN" not in country_codes
        ):
            for code in country_codes:
                collaborations["OTHERS"][code] = (
                    collaborations["OTHERS"].get(code, 0) + 1
                )

    for collaboration, countries in collaborations.items():
        sorted_countries = sorted(countries.items(), key=lambda x: x[1], reverse=True)

        with open(
            f"results/num_papers_and_citations/num_collaborations_{collaboration}.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["Country", "Occurrences"])
            writer.writerows(sorted_countries)

    # endregion

    # region maps

    collaborators = []

    for row in tqdm(df_all_not_only.itertuples()):
        country_list = set(literal_eval(row.countries))
        if len(set(country_list)) < 1:
            continue
        if len(country_list) < 1:
            continue
        updated_list = []
        for country in country_list:
            if country is None:
                continue
            country = "RS" if country == "XK" else country
            country = "CN" if country == "TW" else country
            country = "CN" if country == "HK" else country
            updated_list.append(pycountry.countries.get(alpha_2=country).alpha_3)
        if (
            any(country_code in updated_list for country_code in EU_COUNTRIES_ISO3)
            and len(updated_list) > 1
        ):
            for country in updated_list:
                collaborators.append(
                    (
                        "EU",
                        country,
                    )
                )
                # collaborators.append(("EU", country, int(row.citations),row.type))
        if "USA" in updated_list and len(updated_list) > 1:
            for country in updated_list:
                collaborators.append(
                    (
                        "USA",
                        country,
                    )
                )
                # collaborators.append(("USA", country, int(row.citations),row.type))
        if "CHN" in updated_list and len(updated_list) > 1:
            for country in updated_list:
                collaborators.append(
                    (
                        "CHN",
                        country,
                    )
                )
                # collaborators.append(("CHN", country, int(row.citations),row.type))

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
        if origin == "EU":
            origin_name = "the European Union"
        elif origin == "USA":
            origin_name = "the United States of America"
        elif origin == "CHN":
            origin_name = "China"
        ax.set_title(f"Number of collaborations with {origin_name}")
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
    plt.ylabel("Number of collaborations")
    plt.title("In-house and international collaborations")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration.png"
    )
    plt.close()

    means = means[means["publication_year"] >= 2010]
    sns.lineplot(data=means, x="publication_year", y="count", hue="relation")
    plt.xlabel("Year")
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
    plt.ylabel("Number of collaborations")
    plt.title("International collaborations only")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration_collab.png"
    )
    plt.close()

    means = means[means["publication_year"] >= 2010]
    sns.lineplot(data=means, x="publication_year", y="count", hue="relation")
    plt.xlabel("Year")
    plt.ylabel("Number of collaborations")
    plt.title("International collaborations only")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_collaborations_per_year_per_collaboration_collab_10.png"
    )
    plt.close()

    df = pd.DataFrame(
        collaborations, columns=["relation", "publication_year", "citations"]
    )
    means = (
        df.groupby(["relation", "publication_year"])["citations"]
        .mean()
        .reset_index(name="mean")
    )
    sns.lineplot(data=means, x="publication_year", y="mean", hue="relation")
    plt.xlabel("Year")
    plt.ylabel("Mean citations")
    plt.title("International collaborations only")
    plt.savefig(
        f"results/num_papers_and_citations/lineplot_mean_citations_per_year_per_collaboration.png"
    )
    plt.close()
