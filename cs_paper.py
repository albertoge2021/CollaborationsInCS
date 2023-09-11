from ast import literal_eval
from collections import Counter
from matplotlib import patches
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
eu_df = pd.read_csv("cs_final_eu_changed.csv")
eu_df = eu_df[eu_df["year"] > 1989]
eu_df = eu_df[eu_df["year"] < 2022]
eu_df["citations"].describe().to_csv("paper_results/describe_general_citations.csv")
eu_df[["citations", "type"]].groupby("type").describe().to_csv(
    "paper_results/describe_citations_by_type.csv"
)
unique_collaboration_types = eu_df["type"].unique()
selected_countries = ["US", "CN", "EU"]
colors = ["deepskyblue", "limegreen", "orangered", "mediumpurple"]
Path("paper_results/").mkdir(parents=True, exist_ok=True)

"""us_collaborations = 0
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

for row in tqdm(eu_df.itertuples()):
    country_list = literal_eval(row.countries)
    check = False
    if "EU" in country_list or "CN" in country_list or "US" in country_list:
        check = True
    country_list = set(country_list)
    if check:
        citations = int(row.citations)
        if "US" in country_list:
            us_collaborations_total += 1
            if (
                "US" in country_list
                and "CN" not in country_list
                and "EU" not in country_list
                and len(country_list) == 1
            ):
                us_collaborations += 1
                us_citations += citations
                continue
        if "CN" in country_list:
            cn_collaborations_total += 1
            if (
                "CN" in country_list
                and "US" not in country_list
                and "EU" not in country_list
                and len(country_list) == 1
            ):
                cn_collaborations += 1
                cn_citations += citations
                continue
        if "EU" in country_list:
            eu_collaborations_total += 1
            if (
                "EU" in country_list
                and "US" not in country_list
                and "CN" not in country_list
                and len(country_list) == 1
            ):
                eu_collaborations += 1
                eu_citations += citations
                continue
        if "EU" in country_list and "CN" in country_list and "US" in country_list:
            eu_cn_us_collaborations += 1
            eu_cn_us_citations += citations
            continue
        else:
            if "US" in country_list and "CN" in country_list:
                us_cn_collaborations += 1
                us_cn_citations += citations
            elif "US" in country_list and "EU" in country_list:
                us_eu_collaborations += 1
                us_eu_citations += citations
            elif "EU" in country_list and "CN" in country_list:
                eu_cn_collaborations += 1
                eu_cn_citations += citations

# Define the data
us_data = [us_collaborations, us_eu_collaborations, us_cn_collaborations]
eu_data = [us_eu_collaborations, eu_collaborations, eu_cn_collaborations]
cn_data = [us_cn_collaborations, eu_cn_collaborations, cn_collaborations]
all_data = [eu_cn_us_collaborations, eu_cn_us_collaborations, eu_cn_us_collaborations]

# Define the x-axis labels
labels = ["US", "EU", "CN"]

# Define the x-axis locations for each group of bars
x_us = [0, 1, 2]
x_eu = [5, 6, 7]
x_cn = [10, 11, 12]
x_all = [3, 8, 13]

# Plot the bars
plt.bar(x_us, us_data, color=colors, width=0.8, label="US")
plt.bar(x_eu, eu_data, color=colors, width=0.8, label="EU")
plt.bar(x_cn, cn_data, color=colors, width=0.8, label="CN")
plt.bar(x_all, all_data, color="mediumpurple", width=0.8, label="EU-CN-US")

# Add the x-axis labels and tick marks
plt.xticks([1.5, 6.5, 11.5], labels)
plt.xlabel("Country")
plt.ylabel("Number of Collaborations")

# Create the custom legend
legend_colors = [patches.Patch(color=color) for color in colors]
plt.legend(
    handles=legend_colors,
    labels=["US", "EU", "CN", "US-EU-CN"],
    title="Regions",
    loc="upper left",
)

# Show the plot
plt.title("All publications")
plt.savefig(f"paper_results/bar_country_collaboration_cn_us_eu.png")
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
plt.bar(x_us, us_data, color=colors, width=0.8, label="US")
plt.bar(x_eu, eu_data, color=colors, width=0.8, label="EU")
plt.bar(x_cn, cn_data, color=colors, width=0.8, label="CN")
plt.bar(x_all, all_data, color="mediumpurple", width=0.8, label="EU-CN-US")

# Add the x-axis labels and tick marks
plt.xticks([1.5, 6.5, 11.5], labels)
plt.xlabel("Country")
plt.ylabel("Percentage of Collaborations")

# Add a legend
legend_colors = [patches.Patch(color=color) for color in colors]
plt.legend(
    handles=legend_colors,
    labels=["US", "EU", "CN", "US-EU-CN"],
    title="Regions",
    loc="upper left",
)

# Show the plot
plt.title("All publications")
plt.savefig(f"paper_results/bar_country_collaboration_cn_us_eu_percent.png")
plt.close()

with open(
    f"paper_results/country_collaboration_cn_us_eu_percentage.txt",
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

with open(f"paper_results/country_collaboration_cn_us_eu.txt", "w") as file:
    file.write(f"US - US only collaboration {(us_collaborations)}\n")
    file.write(f"CN - CN only collaboration {(cn_collaborations)}\n")
    file.write(f"EU - EU only collaboration {(eu_collaborations)}\n")
    file.write(f"CN - US collaboration {(us_cn_collaborations)}\n")
    file.write(f"CN - EU collaboration {(eu_cn_collaborations)}\n")
    file.write(f"EU - US collaboration {(us_eu_collaborations)}\n")
    file.write(f"EU - US - CN collaboration {(eu_cn_us_collaborations)}\n")

us_mean_citations = us_citations / us_collaborations if us_collaborations > 0 else 0
eu_mean_citations = eu_citations / eu_collaborations if eu_collaborations > 0 else 0
cn_mean_citations = cn_citations / cn_collaborations if cn_collaborations > 0 else 0
us_eu_mean_citations = (
    us_eu_citations / us_eu_collaborations if us_eu_collaborations > 0 else 0
)
us_cn_mean_citations = (
    us_cn_citations / us_cn_collaborations if us_cn_collaborations > 0 else 0
)
eu_cn_mean_citations = (
    eu_cn_citations / eu_cn_collaborations if eu_cn_collaborations > 0 else 0
)
eu_cn_us_mean_citations = (
    eu_cn_us_citations / eu_cn_us_collaborations if eu_cn_us_collaborations > 0 else 0
)
total_mean_citations = (
    us_citations
    + eu_citations
    + cn_citations
    + us_eu_citations
    + us_cn_citations
    + eu_cn_citations
    + eu_cn_us_citations
)
total_participations = (
    us_collaborations
    + eu_collaborations
    + cn_collaborations
    + us_eu_collaborations
    + us_cn_collaborations
    + eu_cn_collaborations
    + eu_cn_us_collaborations
)

with open(
    "paper_results/country_collaboration_cn_us_eu_citation_mean.txt",
    "w",
) as f:
    f.write(f"US mean citations: {us_mean_citations}\n")
    f.write(f"EU mean citations: {eu_mean_citations}\n")
    f.write(f"CN mean citations: {cn_mean_citations}\n")
    f.write(f"US-EU mean citations: {us_eu_mean_citations}\n")
    f.write(f"US-CN mean citations: {us_cn_mean_citations}\n")
    f.write(f"EU-CN mean citations: {eu_cn_mean_citations}\n")
    f.write(f"EU-CN-US mean citations: {eu_cn_us_mean_citations}\n")
    f.write(f"Mean citations: {total_mean_citations/total_participations}\n")

# Define the data
us_data_means = [us_mean_citations, us_eu_mean_citations, us_cn_mean_citations]
eu_data_means = [us_eu_mean_citations, eu_mean_citations, eu_cn_mean_citations]
cn_data_means = [us_cn_mean_citations, eu_cn_mean_citations, cn_mean_citations]
all_data_means = [
    eu_cn_us_mean_citations,
    eu_cn_us_mean_citations,
    eu_cn_us_mean_citations,
]

# Define the x-axis labels
labels = ["US", "EU", "CN"]

# Define the x-axis locations for each group of bars
# Define the x-axis locations for each group of bars
x_us = [0, 1, 2]
x_eu = [5, 6, 7]
x_cn = [10, 11, 12]
x_all = [3, 8, 13]

# Plot the bars
plt.bar(x_us, us_data_means, color=colors, width=0.8, label="US")
plt.bar(x_eu, eu_data_means, color=colors, width=0.8, label="EU")
plt.bar(x_cn, cn_data_means, color=colors, width=0.8, label="CN")
plt.bar(x_all, all_data_means, color="mediumpurple", width=0.8, label="EU-CN-US")

# Add the x-axis labels and tick marks
plt.xticks([1.5, 6.5, 11.5], labels)
plt.xlabel("Country")
plt.ylabel("Mean Citations")

# Add a legend
legend_colors = [patches.Patch(color=color) for color in colors]
plt.legend(
    handles=legend_colors,
    labels=["US", "EU", "CN", "US-EU-CN"],
    title="Regions",
    loc="upper left",
)

# Show the plot
plt.title("All publications")
plt.savefig(f"paper_results/bar_country_collaboration_citations_cn_us_eu.png")
plt.close()

for collaboration_type in unique_collaboration_types:
    collab_df = eu_df[eu_df["type"] == collaboration_type]
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
        if "EU" in country_list or "CN" in country_list or "US" in country_list:
            check = True
        country_list = set(country_list)
        if check:
            citations = int(row.citations)
            if "US" in country_list:
                us_collaborations_total += 1
            if "CN" in country_list:
                cn_collaborations_total += 1
            if "EU" in country_list:
                eu_collaborations_total += 1
            if "EU" in country_list and "CN" in country_list and "US" in country_list:
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
        f"paper_results/country_collaboration_cn_us_eu_percentage_type_{collaboration_type}.txt",
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
        f"paper_results/country_collaboration_cn_us_eu_type_{collaboration_type}.txt",
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

    # Plot the bars
    plt.bar(x_us, us_data, color=colors, width=0.8, label="US")
    plt.bar(x_eu, eu_data, color=colors, width=0.8, label="EU")
    plt.bar(x_cn, cn_data, color=colors, width=0.8, label="CN")
    plt.bar(x_all, all_data, color="mediumpurple", width=0.8, label="EU-CN-US")

    # Add the x-axis labels and tick marks
    plt.xticks([1.5, 6.5, 11.5], labels)
    plt.xlabel("Country")
    plt.ylabel("Number of Collaborations")

    # Add a legend
    legend_colors = [patches.Patch(color=color) for color in colors]
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
        f"paper_results/bar_country_collaboration_cn_us_eu_type_{collaboration_type}.png"
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
    plt.bar(x_us, us_data, color=colors, width=0.8, label="US")
    plt.bar(x_eu, eu_data, color=colors, width=0.8, label="EU")
    plt.bar(x_cn, cn_data, color=colors, width=0.8, label="CN")
    plt.bar(x_all, all_data, color="mediumpurple", width=0.8, label="EU-CN-US")

    # Add the x-axis labels and tick marks
    plt.xticks([1.5, 6.5, 11.5], labels)
    plt.xlabel("Country")
    plt.ylabel("Percentage of Collaborations")

    # Add a legend
    legend_colors = [patches.Patch(color=color) for color in colors]
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
    # Show the plot
    plt.savefig(
        f"paper_results/bar_country_collaboration_cn_us_eu_percent_type_{collaboration_type}.png"
    )
    plt.close()

    us_mean_citations = us_citations / us_collaborations if us_collaborations > 0 else 0
    eu_mean_citations = eu_citations / eu_collaborations if eu_collaborations > 0 else 0
    cn_mean_citations = cn_citations / cn_collaborations if cn_collaborations > 0 else 0
    us_eu_mean_citations = (
        us_eu_citations / us_eu_collaborations if us_eu_collaborations > 0 else 0
    )
    us_cn_mean_citations = (
        us_cn_citations / us_cn_collaborations if us_cn_collaborations > 0 else 0
    )
    eu_cn_mean_citations = (
        eu_cn_citations / eu_cn_collaborations if eu_cn_collaborations > 0 else 0
    )
    eu_cn_us_mean_citations = (
        eu_cn_us_citations / eu_cn_us_collaborations
        if eu_cn_us_collaborations > 0
        else 0
    )
    total_mean_citations = (
        us_citations
        + eu_citations
        + cn_citations
        + us_eu_citations
        + us_cn_citations
        + eu_cn_citations
        + eu_cn_us_citations
    )
    total_participations = (
        us_collaborations
        + eu_collaborations
        + cn_collaborations
        + us_eu_collaborations
        + us_cn_collaborations
        + eu_cn_collaborations
        + eu_cn_us_collaborations
    )

    with open(
        f"paper_results/country_collaboration_cn_us_eu_citation_mean_type_{collaboration_type}.txt",
        "w",
    ) as f:
        f.write(f"US mean citations: {us_mean_citations}\n")
        f.write(f"EU mean citations: {eu_mean_citations}\n")
        f.write(f"CN mean citations: {cn_mean_citations}\n")
        f.write(f"US-EU mean citations: {us_eu_mean_citations}\n")
        f.write(f"US-CN mean citations: {us_cn_mean_citations}\n")
        f.write(f"EU-CN mean citations: {eu_cn_mean_citations}\n")
        f.write(f"EU-CN-US mean citations: {eu_cn_us_mean_citations}\n")
        f.write(f"Mean citations: {total_mean_citations/total_participations}\n")

    # Define the data
    us_data_means = [us_mean_citations, us_eu_mean_citations, us_cn_mean_citations]
    eu_data_means = [us_eu_mean_citations, eu_mean_citations, eu_cn_mean_citations]
    cn_data_means = [us_cn_mean_citations, eu_cn_mean_citations, cn_mean_citations]
    all_data_means = [
        eu_cn_us_mean_citations,
        eu_cn_us_mean_citations,
        eu_cn_us_mean_citations,
    ]

    # Define the x-axis labels
    labels = ["US", "EU", "CN"]

    # Define the x-axis locations for each group of bars
    # Define the x-axis locations for each group of bars
    x_us = [0, 1, 2]
    x_eu = [5, 6, 7]
    x_cn = [10, 11, 12]
    x_all = [3, 8, 13]

    # Plot the bars
    plt.bar(x_us, us_data_means, color=colors, width=0.8, label="US")
    plt.bar(x_eu, eu_data_means, color=colors, width=0.8, label="EU")
    plt.bar(x_cn, cn_data_means, color=colors, width=0.8, label="CN")
    plt.bar(x_all, all_data_means, color="mediumpurple", width=0.8, label="EU-CN-US")

    # Add the x-axis labels and tick marks
    plt.xticks([1.5, 6.5, 11.5], labels)
    plt.xlabel("Country")
    plt.ylabel("Mean Citations")

    # Add a legend
    legend_colors = [patches.Patch(color=color) for color in colors]
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
        f"paper_results/bar_country_collaboration_citations_cn_us_eu_type_{collaboration_type}.png"
    )
    plt.close()"""

us_ratio_total = []
eu_ratio_total = []
cn_ratio_total = []
us_eu_counts = 0
us_cn_counts = 0
eu_cn_counts = 0
us_citations = 0
eu_citations = 0
cn_citations = 0

for row in tqdm(eu_df.itertuples()):
    us_counts = 0
    eu_counts = 0
    cn_counts = 0
    country_list = literal_eval(row.countries)
    if any(country in country_list for country in country_list):
        num_countries = len(country_list)
        citations = int(row.citations)
        if "US" in country_list:
            us_counts += country_list.count("US")
            us_citations += citations
        if "CN" in country_list:
            cn_counts += country_list.count("CN")
            cn_citations += citations
        if "EU" in country_list:
            eu_counts += country_list.count("EU")
            eu_citations += citations
        if us_counts > 0:
            us_ratio_total.append(
                ((us_counts / num_countries) * 100, citations, row.type, "US")
            )
        if eu_counts > 0:
            eu_ratio_total.append(
                ((eu_counts / num_countries) * 100, citations, row.type, "EU")
            )
        if cn_counts > 0:
            cn_ratio_total.append(
                ((cn_counts / num_countries) * 100, citations, row.type, "CN")
            )

df = pd.DataFrame(us_ratio_total, columns=["ratio", "citations", "type", "country"])

with open(
    "paper_results/correlation_ratio_compared_to_citations_by_type_us.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Pearson test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    df[df["type"] == collaboration_type]["ratio"],
                    df[df["type"] == collaboration_type]["citations"],
                )
            )
            + str(len(df[df["type"] == collaboration_type]))
        )
        f.write("\n")
        f.write(
            "Spearman test for "
            + collaboration_type
            + " - "
            + str(
                stats.spearmanr(
                    df[df["type"] == collaboration_type]["ratio"],
                    df[df["type"] == collaboration_type]["citations"],
                )
            )
            + str(len(df[df["type"] == collaboration_type]))
        )
        f.write("\n")
    f.write(
        "Pearson test general - "
        + str(stats.pearsonr(df["ratio"], df["citations"]))
        + str(len(df.index))
    )
    f.write("\n")
    f.write(
        "Spearman test general - "
        + str(stats.spearmanr(df["ratio"], df["citations"]))
        + str(len(df.index))
    )
    f.write("\n")

sns.lmplot(
    x="ratio",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.xlabel("Participation ratio")
plt.ylabel("Number of Citations")
plt.title("US participation ratio vs citations")
plt.savefig(f"paper_results/scatter_ratio_citations_by_type_us.png")
plt.close()

sns.lmplot(
    x="ratio",
    y="citations",
    data=df,
    scatter=False,
)
plt.xlabel("Participation ratio")
plt.ylabel("Number of Citations")
plt.title("US participation ratio vs citations")
plt.savefig(f"paper_results/scatter_ratio_citations_us.png")
plt.close()

means = df.groupby(["ratio", "type"])["citations"].mean().reset_index(name="mean")
sns.lineplot(data=means, x="ratio", y="mean", hue="type")
plt.xlabel("Participation ratio")
plt.ylabel("Mean citations")
plt.title("US participation ratio vs mean citations")
plt.savefig(f"paper_results/scatter_mean_citations_by_ratio_by_type_us.png")
plt.close()

df = pd.DataFrame(eu_ratio_total, columns=["ratio", "citations", "type", "country"])

with open(
    "paper_results/correlation_ratio_compared_to_citations_by_type_eu.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Pearson test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    df[df["type"] == collaboration_type]["ratio"],
                    df[df["type"] == collaboration_type]["citations"],
                )
            )
            + str(len(df[df["type"] == collaboration_type]))
        )
        f.write("\n")
        f.write(
            "Spearman test for "
            + collaboration_type
            + " - "
            + str(
                stats.spearmanr(
                    df[df["type"] == collaboration_type]["ratio"],
                    df[df["type"] == collaboration_type]["citations"],
                )
            )
            + str(len(df[df["type"] == collaboration_type]))
        )
        f.write("\n")
    f.write(
        "Pearson test general - "
        + str(stats.pearsonr(df["ratio"], df["citations"]))
        + str(len(df.index))
    )
    f.write("\n")
    f.write(
        "Spearman test general - "
        + str(stats.spearmanr(df["ratio"], df["citations"]))
        + str(len(df.index))
    )
    f.write("\n")

sns.lmplot(
    x="ratio",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.xlabel("Participation ratio")
plt.ylabel("Number of Citations")
plt.title("EU participation ratio vs citations")
plt.savefig(f"paper_results/scatter_ratio_citations_by_type_eu.png")
plt.close()

sns.lmplot(
    x="ratio",
    y="citations",
    data=df,
    scatter=False,
)
plt.xlabel("Participation ratio")
plt.ylabel("Number of Citations")
plt.title("EU participation ratio vs citations")
plt.savefig(f"paper_results/scatter_ratio_citations_eu.png")
plt.close()

means = df.groupby(["ratio", "type"])["citations"].mean().reset_index(name="mean")
sns.lineplot(data=means, x="ratio", y="mean", hue="type")
plt.xlabel("Participation ratio")
plt.ylabel("Mean citations")
plt.title("EU participation ratio vs mean citations")
plt.savefig(f"paper_results/scatter_mean_citations_by_ratio_by_type_eu.png")
plt.close()

df = pd.DataFrame(cn_ratio_total, columns=["ratio", "citations", "type", "country"])

with open(
    "paper_results/correlation_ratio_compared_to_citations_by_type_cn.txt",
    "w",
) as f:
    for collaboration_type in unique_collaboration_types:
        f.write(
            "Pearson test for "
            + collaboration_type
            + " - "
            + str(
                stats.pearsonr(
                    df[df["type"] == collaboration_type]["ratio"],
                    df[df["type"] == collaboration_type]["citations"],
                )
            )
            + str(len(df[df["type"] == collaboration_type]))
        )
        f.write("\n")
        f.write(
            "Spearman test for "
            + collaboration_type
            + " - "
            + str(
                stats.spearmanr(
                    df[df["type"] == collaboration_type]["ratio"],
                    df[df["type"] == collaboration_type]["citations"],
                )
            )
            + str(len(df[df["type"] == collaboration_type]))
        )
        f.write("\n")
    f.write(
        "Pearson test general - "
        + str(stats.pearsonr(df["ratio"], df["citations"]))
        + str(len(df.index))
    )
    f.write("\n")
    f.write(
        "Spearman test general - "
        + str(stats.spearmanr(df["ratio"], df["citations"]))
        + str(len(df.index))
    )
    f.write("\n")

sns.lmplot(
    x="ratio",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.xlabel("Participation ratio")
plt.ylabel("Number of Citations")
plt.title("CN participation ratio vs citations")
plt.savefig(f"paper_results/scatter_ratio_citations_by_type_cn.png")
plt.close()

sns.lmplot(
    x="ratio",
    y="citations",
    data=df,
    scatter=False,
)
plt.xlabel("Participation ratio")
plt.ylabel("Number of Citations")
plt.title("CN participation ratio vs citations")
plt.savefig(f"paper_results/scatter_ratio_citations_cn.png")
plt.close()

means = df.groupby(["ratio", "type"])["citations"].mean().reset_index(name="mean")
sns.lineplot(data=means, x="ratio", y="mean", hue="type")
plt.xlabel("Participation ratio")
plt.ylabel("Mean citations")
plt.title("CN participation ratio vs mean citations")
plt.savefig(f"paper_results/scatter_mean_citations_by_ratio_by_type_cn.png")
plt.close()

all_countries_list = cn_ratio_total + eu_ratio_total + us_ratio_total

df = pd.DataFrame(all_countries_list, columns=["ratio", "citations", "type", "country"])
sns.lmplot(
    x="ratio",
    y="citations",
    hue="country",
    data=df,
    scatter=False,
    palette=["orangered", "limegreen", "deepskyblue", "mediumpurple"],
)
plt.xlabel("Participation ratio")
plt.ylabel("Number of Citations")
plt.title("Participation ratio - Citations")
plt.savefig(f"paper_results/scatter_ratio_citations_by_countries.png")
plt.close()

df.boxplot(by="country", column=["citations"], grid=False)
plt.title("Average number of citations by Region")
plt.savefig(f"paper_results/boxplot_citations_by_countries.png")
plt.close()
"""
collaborations = []

for row in tqdm(eu_df.itertuples()):
    country_list = literal_eval(row.countries)
    check = False
    if "EU" in country_list or "CN" in country_list or "US" in country_list:
        check = True
    country_list = set(country_list)
    if check:
        citations = int(row.citations)
        if "EU" in country_list and "CN" in country_list and "US" in country_list:
            collaborations.append(("EU-CN-US", int(row.year), citations))
            continue
        elif "US" in country_list and "CN" in country_list:
            collaborations.append(("US-CN", int(row.year), citations))
            continue
        elif "US" in country_list and "EU" in country_list:
            collaborations.append(("US-EU", int(row.year), citations))
            continue
        elif "EU" in country_list and "CN" in country_list:
            collaborations.append(("EU-CN", int(row.year), citations))
            continue
        elif (
            "US" in country_list
            and "CN" not in country_list
            and "EU" not in country_list
            and len(country_list) == 1
        ):
            collaborations.append(("US", int(row.year), citations))
            continue
        elif (
            "CN" in country_list
            and "US" not in country_list
            and "EU" not in country_list
            and len(country_list) == 1
        ):
            collaborations.append(("CN", int(row.year), citations))
            continue
        elif (
            "EU" in country_list
            and "US" not in country_list
            and "CN" not in country_list
            and len(country_list) == 1
        ):
            collaborations.append(("EU", int(row.year), citations))
            continue

df = pd.DataFrame(collaborations, columns=["relation", "year", "citations"])
means = df.groupby(["relation", "year"]).size().reset_index(name="count")
sns.lineplot(data=means, x="year", y="count", hue="relation")
plt.xlabel("Year")
plt.ylabel("Number of collaborations")
plt.title("In-house and international collaborations")
plt.savefig(f"paper_results/lineplot_collaborations_per_year_per_collaboration.png")
plt.close()

means = means[means["year"] >= 2010]
sns.lineplot(data=means, x="year", y="count", hue="relation")
plt.xlabel("Year")
plt.ylabel("Number of collaborations")
plt.title("In-house and international collaborations")
plt.savefig(f"paper_results/lineplot_collaborations_per_year_per_collaboration_10.png")
plt.close()

means = df.groupby(["relation", "year"]).size().reset_index(name="count")
means = means[~means["relation"].isin(["CN", "US", "EU"])]
sns.lineplot(data=means, x="year", y="count", hue="relation")
plt.xlabel("Year")
plt.ylabel("Number of collaborations")
plt.title("International collaborations only")
plt.savefig(
    f"paper_results/lineplot_collaborations_per_year_per_collaboration_collab.png"
)
plt.close()

means = means[means["year"] >= 2010]
sns.lineplot(data=means, x="year", y="count", hue="relation")
plt.xlabel("Year")
plt.ylabel("Number of collaborations")
plt.title("International collaborations only")
plt.savefig(
    f"paper_results/lineplot_collaborations_per_year_per_collaboration_collab_10.png"
)
plt.close()

df = pd.DataFrame(collaborations, columns=["relation", "year", "citations"])
means = df.groupby(["relation", "year"])["citations"].mean().reset_index(name="mean")
sns.lineplot(data=means, x="year", y="mean", hue="relation")
plt.xlabel("Year")
plt.ylabel("Mean citations")
plt.title("International collaborations only")
plt.savefig(f"paper_results/lineplot_mean_citations_per_year_per_collaboration.png")
plt.close()

collaborations = []

for row in tqdm(eu_df.itertuples()):
    country_list = literal_eval(row.countries)
    check = False
    if "EU" in country_list or "CN" in country_list or "US" in country_list:
        check = True
    country_list = set(country_list)
    if check:
        citations = int(row.citations)
        if "EU" in country_list and "CN" in country_list and "US" in country_list:
            for topic in literal_eval(row.concepts):
                collaborations.append(("CN-EU-US", int(row.year), citations, topic))
        elif "US" in country_list and "CN" in country_list:
            for topic in literal_eval(row.concepts):
                collaborations.append(("CN-US", int(row.year), citations, topic))
        elif "US" in country_list and "EU" in country_list:
            for topic in literal_eval(row.concepts):
                collaborations.append(("EU-US", int(row.year), citations, topic))
        elif "EU" in country_list and "CN" in country_list:
            for topic in literal_eval(row.concepts):
                collaborations.append(("CN-EU", int(row.year), citations, topic))
        elif (
            "US" in country_list
            and "CN" not in country_list
            and "EU" not in country_list
            and len(country_list) == 1
        ):
            for topic in literal_eval(row.concepts):
                collaborations.append(("US", int(row.year), citations, topic))
        elif (
            "CN" in country_list
            and "US" not in country_list
            and "EU" not in country_list
            and len(country_list) == 1
        ):
            for topic in literal_eval(row.concepts):
                collaborations.append(("CN", int(row.year), citations, topic))
        elif (
            "EU" in country_list
            and "US" not in country_list
            and "CN" not in country_list
            and len(country_list) == 1
        ):
            for topic in literal_eval(row.concepts):
                collaborations.append(("EU", int(row.year), citations, topic))

df = pd.DataFrame(collaborations, columns=["relation", "year", "citations", "concept"])
df[["relation", "citations"]].groupby("relation").describe().to_csv(
    "paper_results/describe_citations_by_relation.csv"
)
unique_relation_types = df["relation"].unique()

for relation in unique_relation_types:
    relation_df = df[df["relation"] == relation]
    test = (
        relation_df.groupby("concept")
        .size()
        .reset_index(name="count")
        .sort_values(by=["count"], ascending=False)
        .head(11)
    )
    test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
    new_df = relation_df.loc[relation_df["concept"].isin(test.concept.to_list())]
    means_full = new_df.groupby(["concept", "year"]).size().reset_index(name="count")
    sns.lineplot(
        data=means_full, x="year", y="count", hue="concept", markers=True, sort=True
    )
    plt.xlabel("Year")
    plt.legend(title="Concept")
    plt.ylabel("Number of collaborations")
    plt.title("10 most common topics by year for " + relation)
    plt.savefig(f"paper_results/line_topics_by_year_{relation}.png")
    plt.close()
"""
