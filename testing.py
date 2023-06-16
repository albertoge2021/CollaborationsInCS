from ast import literal_eval
from collections import Counter
import pandas as pd
import warnings
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

# Setup Data
eu_df = pd.read_csv("cs_eu.csv")
unique_collaboration_types = eu_df["type"].unique()
selected_countries = ["US", "CN", "EU"]

eu_topics = []
us_eu_topics = []

for row in tqdm(eu_df.itertuples()):
    locations = literal_eval(row.location)
    country_list = []
    for location in locations:
        country_code = location["country"]
        if country_code in selected_countries:
            country_list.append(country_code)
    concepts = literal_eval(row.concepts)
    country_list = set(country_list)
    if "EU" in country_list:
        if (
            "EU" in country_list
            and "US" not in country_list
            and "CN" not in country_list
            and len(country_list) == 1
        ):
            for concept in concepts:
                eu_topics.append((concept, row.year, row.type))
            continue
        else:
            if (
                "US" in country_list
                and "EU" in country_list
                and not "CN" in country_list
            ):
                for concept in concepts:
                    us_eu_topics.append((concept, row.year, row.type))

df = pd.DataFrame(eu_topics, columns=["concept", "year", "type"])

us_topics = []

for row in tqdm(eu_df.itertuples()):
    locations = literal_eval(row.location)
    country_list = []
    for location in locations:
        country_code = location["country"]
        if country_code in selected_countries:
            country_list.append(country_code)
    concepts = literal_eval(row.concepts)
    country_list = set(country_list)
    if "US" in country_list:
        if (
            "US" in country_list
            and "CN" not in country_list
            and "EU" not in country_list
            and len(country_list) == 1
        ):
            for concept in concepts:
                us_topics.append((concept, row.year, row.type))
            continue

df = pd.DataFrame(us_topics, columns=["concept", "year", "type"])
df = df[df["concept"] != "Computer science"]

# Group the dataframe by year and concept, and count the number of occurrences of each concept in each year
grouped = df.groupby(["year", "concept"])["concept"].count()

# Sort the resulting dataframe by year and the count of occurrences, in descending order
sorted_df = grouped.reset_index(name="count").sort_values(
    ["year", "count"], ascending=[True, False]
)

# Use the groupby function to group the dataframe by year, and select the top ten concepts with the highest number of occurrences in each year using the head() function
top10_df = sorted_df.groupby("year").head(10)

# Pivot the dataframe to create a matrix where the rows correspond to years and the columns correspond to concepts
pivot_df = top10_df.pivot(index="year", columns="concept", values="count")

# Create a line plot of the pivot dataframe, showing the number of occurrences of each concept over time
pivot_df.plot(kind="line", figsize=(12, 8))

# Set the axis labels and title
plt.xlabel("Year")
plt.ylabel("Number of Occurrences")
plt.title("Top 10 Most Repeated Concepts by Year")
plt.savefig(f"computer_science/topic_analysis/line_topics_second_by_year_us.png")
plt.close()

cn_topics = []
eu_cn_topics = []
cn_us_topics = []
cn_eu_us_topics = []

for row in tqdm(eu_df.itertuples()):
    locations = literal_eval(row.location)
    country_list = []
    for location in locations:
        country_code = location["country"]
        if country_code in selected_countries:
            country_list.append(country_code)
    concepts = literal_eval(row.concepts)
    country_list = set(country_list)
    if "CN" in country_list:
        if (
            "CN" in country_list
            and "US" not in country_list
            and "EU" not in country_list
            and len(country_list) == 1
        ):
            for concept in concepts:
                cn_topics.append((concept, row.year, row.type))
            continue
        else:
            if (
                "US" in country_list
                and "CN" in country_list
                and not "EU" in country_list
            ):
                for concept in concepts:
                    cn_us_topics.append((concept, row.year, row.type))
            if (
                "EU" in country_list
                and "CN" in country_list
                and not "US" in country_list
            ):
                for concept in concepts:
                    eu_cn_topics.append((concept, row.year, row.type))
    if "EU" in country_list and "CN" in country_list and "US" in country_list:
        for concept in concepts:
            cn_eu_us_topics.append((concept, row.year, row.type))
        continue


df = pd.DataFrame(cn_topics, columns=["concept", "year", "type"])

df = df[df["concept"] != "Computer science"]

# Group the dataframe by year and concept, and count the number of occurrences of each concept in each year
grouped = df.groupby(["year", "concept"])["concept"].count()

# Sort the resulting dataframe by year and the count of occurrences, in descending order
sorted_df = grouped.reset_index(name="count").sort_values(
    ["year", "count"], ascending=[True, False]
)

# Use the groupby function to group the dataframe by year, and select the top ten concepts with the highest number of occurrences in each year using the head() function
top10_df = sorted_df.groupby("year").head(10)

# Pivot the dataframe to create a matrix where the rows correspond to years and the columns correspond to concepts
pivot_df = top10_df.pivot(index="year", columns="concept", values="count")

# Create a line plot of the pivot dataframe, showing the number of occurrences of each concept over time
pivot_df.plot(kind="line", figsize=(12, 8))

# Set the axis labels and title
plt.xlabel("Year")
plt.ylabel("Number of Occurrences")
plt.title("Top 10 Most Repeated Concepts by Year")
plt.savefig(f"computer_science/topic_analysis/line_topics_second_by_year_cn.png")
plt.close()

df = pd.DataFrame(cn_eu_us_topics, columns=["concept", "year", "type"])

df = df[df["concept"] != "Computer science"]

# Group the dataframe by year and concept, and count the number of occurrences of each concept in each year
grouped = df.groupby(["year", "concept"])["concept"].count()

# Sort the resulting dataframe by year and the count of occurrences, in descending order
sorted_df = grouped.reset_index(name="count").sort_values(
    ["year", "count"], ascending=[True, False]
)

# Use the groupby function to group the dataframe by year, and select the top ten concepts with the highest number of occurrences in each year using the head() function
top10_df = sorted_df.groupby("year").head(10)

# Pivot the dataframe to create a matrix where the rows correspond to years and the columns correspond to concepts
pivot_df = top10_df.pivot(index="year", columns="concept", values="count")

# Create a line plot of the pivot dataframe, showing the number of occurrences of each concept over time
pivot_df.plot(kind="line", figsize=(12, 8))

# Set the axis labels and title
plt.xlabel("Year")
plt.ylabel("Number of Occurrences")
plt.title("Top 10 Most Repeated Concepts by Year")
plt.savefig(f"computer_science/topic_analysis/line_topics_second_by_year_cn_us_eu.png")
plt.close()

df = pd.DataFrame(cn_us_topics, columns=["concept", "year", "type"])

df = df[df["concept"] != "Computer science"]

# Group the dataframe by year and concept, and count the number of occurrences of each concept in each year
grouped = df.groupby(["year", "concept"])["concept"].count()

# Sort the resulting dataframe by year and the count of occurrences, in descending order
sorted_df = grouped.reset_index(name="count").sort_values(
    ["year", "count"], ascending=[True, False]
)

# Use the groupby function to group the dataframe by year, and select the top ten concepts with the highest number of occurrences in each year using the head() function
top10_df = sorted_df.groupby("year").head(10)

# Pivot the dataframe to create a matrix where the rows correspond to years and the columns correspond to concepts
pivot_df = top10_df.pivot(index="year", columns="concept", values="count")

# Create a line plot of the pivot dataframe, showing the number of occurrences of each concept over time
pivot_df.plot(kind="line", figsize=(12, 8))

# Set the axis labels and title
plt.xlabel("Year")
plt.ylabel("Number of Occurrences")
plt.title("Top 10 Most Repeated Concepts by Year")
plt.savefig(f"computer_science/topic_analysis/line_topics_second_by_year_cn_us.png")
plt.close()

df = pd.DataFrame(eu_cn_topics, columns=["concept", "year", "type"])

df = df[df["concept"] != "Computer science"]

# Group the dataframe by year and concept, and count the number of occurrences of each concept in each year
grouped = df.groupby(["year", "concept"])["concept"].count()

# Sort the resulting dataframe by year and the count of occurrences, in descending order
sorted_df = grouped.reset_index(name="count").sort_values(
    ["year", "count"], ascending=[True, False]
)

# Use the groupby function to group the dataframe by year, and select the top ten concepts with the highest number of occurrences in each year using the head() function
top10_df = sorted_df.groupby("year").head(10)

# Pivot the dataframe to create a matrix where the rows correspond to years and the columns correspond to concepts
pivot_df = top10_df.pivot(index="year", columns="concept", values="count")

# Create a line plot of the pivot dataframe, showing the number of occurrences of each concept over time
pivot_df.plot(kind="line", figsize=(12, 8))

# Set the axis labels and title
plt.xlabel("Year")
plt.ylabel("Number of Occurrences")
plt.title("Top 10 Most Repeated Concepts by Year")
plt.savefig(f"computer_science/topic_analysis/line_topics_second_by_year_cn_eu.png")
plt.close()

eu_topics = []
us_eu_topics = []

for row in tqdm(eu_df.itertuples()):
    locations = literal_eval(row.location)
    country_list = []
    for location in locations:
        country_code = location["country"]
        if country_code in selected_countries:
            country_list.append(country_code)
    concepts = literal_eval(row.concepts)
    country_list = set(country_list)
    if "EU" in country_list:
        if (
            "EU" in country_list
            and "US" not in country_list
            and "CN" not in country_list
            and len(country_list) == 1
        ):
            for concept in concepts:
                eu_topics.append((concept, row.year, row.type))
            continue
        else:
            if (
                "US" in country_list
                and "EU" in country_list
                and not "CN" in country_list
            ):
                for concept in concepts:
                    us_eu_topics.append((concept, row.year, row.type))

df = pd.DataFrame(eu_topics, columns=["concept", "year", "type"])

df = df[df["concept"] != "Computer science"]

# Group the dataframe by year and concept, and count the number of occurrences of each concept in each year
grouped = df.groupby(["year", "concept"])["concept"].count()

# Sort the resulting dataframe by year and the count of occurrences, in descending order
sorted_df = grouped.reset_index(name="count").sort_values(
    ["year", "count"], ascending=[True, False]
)

# Use the groupby function to group the dataframe by year, and select the top ten concepts with the highest number of occurrences in each year using the head() function
top10_df = sorted_df.groupby("year").head(10)

# Pivot the dataframe to create a matrix where the rows correspond to years and the columns correspond to concepts
pivot_df = top10_df.pivot(index="year", columns="concept", values="count")

# Create a line plot of the pivot dataframe, showing the number of occurrences of each concept over time
pivot_df.plot(kind="line", figsize=(12, 8))

# Set the axis labels and title
plt.xlabel("Year")
plt.ylabel("Number of Occurrences")
plt.title("Top 10 Most Repeated Concepts by Year")
plt.savefig(f"computer_science/topic_analysis/line_topics_second_by_year_eu.png")
plt.close()


df = pd.DataFrame(us_eu_topics, columns=["concept", "year", "type"])
df = df[df["concept"] != "Computer science"]

# Group the dataframe by year and concept, and count the number of occurrences of each concept in each year
grouped = df.groupby(["year", "concept"])["concept"].count()

# Sort the resulting dataframe by year and the count of occurrences, in descending order
sorted_df = grouped.reset_index(name="count").sort_values(
    ["year", "count"], ascending=[True, False]
)

# Use the groupby function to group the dataframe by year, and select the top ten concepts with the highest number of occurrences in each year using the head() function
top10_df = sorted_df.groupby("year").head(10)

# Pivot the dataframe to create a matrix where the rows correspond to years and the columns correspond to concepts
pivot_df = top10_df.pivot(index="year", columns="concept", values="count")

# Create a line plot of the pivot dataframe, showing the number of occurrences of each concept over time
pivot_df.plot(kind="line", figsize=(12, 8))

# Set the axis labels and title
plt.xlabel("Year")
plt.ylabel("Number of Occurrences")
plt.title("Top 10 Most Repeated Concepts by Year")
plt.savefig(f"computer_science/topic_analysis/line_topics_second_by_year_us_eu.png")
plt.close()
