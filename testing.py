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
df = pd.read_csv("cs_mean.csv")
unique_collaboration_types = df["type"].unique()

# Define the countries of interest
for collaboration_type in unique_collaboration_types:
    df = pd.read_csv("cs_mean.csv")
    collabs = {"NA": [], "EU": [], "AS": [], "OC": [], "SA": [], "AF": []}
    df = df[df["type"] == collaboration_type]
    for row in tqdm(df.itertuples()):
        locations = literal_eval(row.location)
        continent_list= []
        for continent in locations:
            continent_list.append(continent["continent"])
        for i in range(len(continent_list)):
            for j in range(i+1, len(continent_list)):
                collabs[continent_list[i]] += [continent_list[j]]
                collabs[continent_list[j]] += [continent_list[i]]
    new_df = pd.DataFrame({"continent": [], "collaboration": [], "number": int})
    for k, v in collabs.items():
        values = Counter(v)
        for key, value in values.items():
            new_df = new_df.append(
                pd.Series(
                    [k, key, value],
                    index=new_df.columns,
                ),
                ignore_index=True,
            )

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
    new_df.groupby(["continent", "collaboration"]).sum().unstack().plot.pie(
        subplots=True,
        autopct="%1.1f%%",
        legend=False,
        startangle=90,
        figsize=(10, 7),
        layout=(-1, 3),
    )
    fig.suptitle("Collaboration by Continent", fontsize=16, fontweight="bold")
    plt.savefig(f'computer_science/continent_analysis/pie_test_continent_collaboration_by_type_{collaboration_type}.png')
    plt.close()

df = pd.read_csv("cs_eu.csv")
for collaboration_type in unique_collaboration_types:
    df = pd.read_csv("cs_eu.csv")
    new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
    collabs = {"EU": [], "US": [], "CN": []}
    df = df[df["type"] == collaboration_type]
    for row in tqdm(df.itertuples()):
        locations = literal_eval(row.location)
        country_list = []
        for location in locations:
            country_list.append(location["country"])
        for i in range(len(country_list)):
            for j in range(i+1, len(country_list)):
                if (country_list[i] in collabs.keys()) and (country_list[j] in collabs.keys()):
                    collabs[country_list[i]] += [country_list[j]]
                    collabs[country_list[j]] += [country_list[i]]
    for k, v in collabs.items():
        values = Counter(v)
        for key, value in values.items():
            new_df = new_df.append(
                pd.Series(
                    [k, key, value],
                    index=new_df.columns,
                ),
                ignore_index=True,
            )

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
    new_df.groupby(["country", "collaboration"]).sum().unstack().plot.pie(
        subplots=True,
        autopct="%1.1f%%",
        legend=False,
        startangle=90,
        figsize=(10, 7),
        layout=(-1, 3),
    )
    fig.suptitle("Collaboration by Country", fontsize=16, fontweight="bold")
    plt.savefig(f'computer_science/country_analysis/pie_country_collaboration_by_type_{collaboration_type}.png')
    plt.close()

countries = ["EU", "US", "CN"]
for collaboration_type in unique_collaboration_types:
    df = pd.read_csv("cs_eu.csv")
    new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
    collabs = {country: [] for country in countries}
    df = df[df["type"] == collaboration_type]
    for row in tqdm(df.itertuples()):
        locations = literal_eval(row.location)
        country_list= []
        for location in locations:
            country_code = location["country"]
            if country_code in countries:
                country_list.append(country_code)
        for i in range(len(country_list)):
            for j in range(i+1, len(country_list)):
                collabs[country_list[i]] += [country_list[j]]
                collabs[country_list[j]] += [country_list[i]]
    for k, v in collabs.items():
        values = Counter(v)
        for key, value in values.items():
            new_df = new_df.append(
                pd.Series(
                    [k, key, value],
                    index=new_df.columns,
                ),
                ignore_index=True,
            )
    new_df.groupby(["country", "collaboration"]).sum().unstack().plot(kind='bar',y='number')
    plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_by_type_{collaboration_type}.png')
    plt.close()

countries = ["EU", "US", "CN"]

for collaboration_type in unique_collaboration_types:
    us_collaborations = 0
    eu_collaborations = 0
    cn_collaborations = 0
    us_eu_collaborations = 0
    us_cn_collaborations = 0
    eu_cn_collaborations = 0
    eu_cn_us_collaborations = 0
    df = pd.read_csv("cs_eu.csv")
    df = df[df["type"] == collaboration_type]
    for row in tqdm(df.itertuples()):
        locations = literal_eval(row.location)
        country_list= []
        for location in locations:
            country_code = location["country"]
            if country_code in countries:
                country_list.append(country_code)
        if "US" in set(country_list):
            us_collaborations += 1
            if "CN" in country_list and "EU" in country_list:
                eu_cn_us_collaborations += 1
            if "CN" in country_list:
                us_cn_collaborations += 1
            if "EU" in country_list:
                us_eu_collaborations += 1
        if "CN" in country_list:
            cn_collaborations += 1
            if "EU" in country_list:
                eu_cn_collaborations += 1
        if "EU" in country_list:
            eu_collaborations += 1
    with open(f"computer_science/country_analysis/country_collaboration_cn_us_eu_percentage_by_type_{collaboration_type}.txt", "w") as file:
        file.write(f"{collaboration_type}\n")
        file.write(f"CN - US collaboration represents {(us_cn_collaborations / us_collaborations) * 100:.2f}% of total US collaborations\n")
        file.write(f"CN - US collaboration represents {(us_cn_collaborations / cn_collaborations) * 100:.2f}% of total Chinese collaborations\n")
        file.write(f"CN - EU collaboration represents {(eu_cn_collaborations / cn_collaborations) * 100:.2f}% of total CN collaborations\n")
        file.write(f"CN - EU collaboration represents {(eu_cn_collaborations / eu_collaborations) * 100:.2f}% of total EU collaborations\n")
        file.write(f"EU - US collaboration represents {(us_eu_collaborations / us_collaborations) * 100:.2f}% of total US collaborations\n")
        file.write(f"EU - US collaboration represents {(us_eu_collaborations / eu_collaborations) * 100:.2f}% of total EU collaborations\n")

    with open(f"computer_science/country_analysis/country_collaboration_cn_us_eu_by_type_{collaboration_type}.txt", "w") as file:
        file.write(f"{collaboration_type}\n")
        file.write(f"CN - US collaboration {(us_cn_collaborations)}\n")
        file.write(f"CN - EU collaboration {(eu_cn_collaborations)}\n")
        file.write(f"EU - US collaboration {(us_eu_collaborations)}\n")
        file.write(f"EU - US - CN collaboration {(eu_cn_us_collaborations)}\n")


    # Define the data
    us_data = [us_collaborations, us_eu_collaborations, us_cn_collaborations]
    eu_data = [us_eu_collaborations, eu_collaborations, eu_cn_collaborations ]
    cn_data = [eu_cn_collaborations, us_cn_collaborations, cn_collaborations]

    # Define the x-axis labels
    labels = ['US Collaborations', 'EU Collaborations', 'CN Collaborations']

    # Define the x-axis locations for each group of bars
    x_us = [0, 4, 8]
    x_eu = [1, 5, 9]
    x_cn = [2, 6, 10]

    # Plot the bars
    plt.bar(x_us, us_data, color='blue', width=0.8, label='US')
    plt.bar(x_eu, eu_data, color='red', width=0.8, label='EU')
    plt.bar(x_cn, cn_data, color='green', width=0.8, label='CN')

    # Add the x-axis labels and tick marks
    plt.xticks([1.5, 5.5, 9.5], labels)
    plt.xlabel('Collaboration Type')
    plt.ylabel('Number of Collaborations')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_cn_us_eu_by_type_{collaboration_type}.png')
    plt.close()

    # Define the data
    us_data = [(us_collaborations / us_collaborations) * 100, (us_eu_collaborations / us_collaborations) * 100, (us_cn_collaborations / us_collaborations) * 100]
    eu_data = [(us_eu_collaborations / eu_collaborations) * 100, (eu_collaborations / eu_collaborations) * 100, (us_cn_collaborations / eu_collaborations) * 100 ]
    cn_data = [(us_cn_collaborations / cn_collaborations) * 100, (eu_cn_collaborations / cn_collaborations) * 100, (cn_collaborations / cn_collaborations) * 100]

    # Define the x-axis labels
    labels = ['US Collaborations', 'EU Collaborations', 'CN Collaborations']

    # Define the x-axis locations for each group of bars
    x_us = [0, 4, 8]
    x_eu = [1, 5, 9]
    x_cn = [2, 6, 10]

    # Plot the bars
    plt.bar(x_us, us_data, color='blue', width=0.8, label='US')
    plt.bar(x_eu, eu_data, color='red', width=0.8, label='EU')
    plt.bar(x_cn, cn_data, color='green', width=0.8, label='CN')

    # Add the x-axis labels and tick marks
    plt.xticks([1.5, 5.5, 9.5], labels)
    plt.xlabel('Collaboration Type')
    plt.ylabel('Number of Collaborations')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_cn_us_eu_percent_by_type_{collaboration_type}.png')
    plt.close()

# Define the countries of interest
df = pd.read_csv("cs_mean.csv")
collabs = {"NA": [], "EU": [], "AS": [], "OC": [], "SA": [], "AF": []}
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    continent_list= []
    for continent in locations:
        continent_list.append(continent["continent"])
    for i in range(len(continent_list)):
        for j in range(i+1, len(continent_list)):
            collabs[continent_list[i]] += [continent_list[j]]
            collabs[continent_list[j]] += [continent_list[i]]
new_df = pd.DataFrame({"continent": [], "collaboration": [], "number": int})
for k, v in collabs.items():
    values = Counter(v)
    for key, value in values.items():
        new_df = new_df.append(
            pd.Series(
                [k, key, value],
                index=new_df.columns,
            ),
            ignore_index=True,
        )

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
new_df.groupby(["continent", "collaboration"]).sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
    layout=(-1, 3),
)
fig.suptitle("Collaboration by Continent", fontsize=16, fontweight="bold")
plt.savefig(f'computer_science/continent_analysis/pie_test_continent_collaboration_total.png')
plt.close()

df = pd.read_csv("cs_eu.csv")
new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
collabs = {"EU": [], "US": [], "CN": []}
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    country_list = []
    for location in locations:
        country_list.append(location["country"])
    for i in range(len(country_list)):
        for j in range(i+1, len(country_list)):
            if (country_list[i] in collabs.keys()) and (country_list[j] in collabs.keys()):
                collabs[country_list[i]] += [country_list[j]]
                collabs[country_list[j]] += [country_list[i]]
for k, v in collabs.items():
    values = Counter(v)
    for key, value in values.items():
        new_df = new_df.append(
            pd.Series(
                [k, key, value],
                index=new_df.columns,
            ),
            ignore_index=True,
        )

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
new_df.groupby(["country", "collaboration"]).sum().unstack().plot.pie(
    subplots=True,
    autopct="%1.1f%%",
    legend=False,
    startangle=90,
    figsize=(10, 7),
    layout=(-1, 3),
)
fig.suptitle("Collaboration by Country", fontsize=16, fontweight="bold")
plt.savefig(f'computer_science/country_analysis/pie_country_collaboration_total.png')
plt.close()

countries = ["EU", "US", "CN"]
df = pd.read_csv("cs_eu.csv")
new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
collabs = {country: [] for country in countries}
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    country_list= []
    for location in locations:
        country_code = location["country"]
        if country_code in countries:
            country_list.append(country_code)
    for i in range(len(country_list)):
        for j in range(i+1, len(country_list)):
            collabs[country_list[i]] += [country_list[j]]
            collabs[country_list[j]] += [country_list[i]]
for k, v in collabs.items():
    values = Counter(v)
    for key, value in values.items():
        new_df = new_df.append(
            pd.Series(
                [k, key, value],
                index=new_df.columns,
            ),
            ignore_index=True,
        )
new_df.groupby(["country", "collaboration"]).sum().unstack().plot(kind='bar',y='number')
plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_total.png')
plt.close()

countries = ["EU", "US", "CN"]

us_collaborations = 0
eu_collaborations = 0
cn_collaborations = 0
us_eu_collaborations = 0
us_cn_collaborations = 0
eu_cn_collaborations = 0
eu_cn_us_collaborations = 0
df = pd.read_csv("cs_eu.csv")
df = df[df["type"] == collaboration_type]
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    country_list= []
    for location in locations:
        country_code = location["country"]
        if country_code in countries:
            country_list.append(country_code)
    if "US" in set(country_list):
        us_collaborations += 1
        if "CN" in country_list and "EU" in country_list:
            eu_cn_us_collaborations += 1
        if "CN" in country_list:
            us_cn_collaborations += 1
        if "EU" in country_list:
            us_eu_collaborations += 1
    if "CN" in country_list:
        cn_collaborations += 1
        if "EU" in country_list:
            eu_cn_collaborations += 1
    if "EU" in country_list:
        eu_collaborations += 1
with open("computer_science/country_analysis/country_collaboration_cn_us_eu_percentage_total.txt", "w") as file:
    file.write(f"{collaboration_type}\n")
    file.write(f"CN - US collaboration represents {(us_cn_collaborations / us_collaborations) * 100:.2f}% of total US collaborations\n")
    file.write(f"CN - US collaboration represents {(us_cn_collaborations / cn_collaborations) * 100:.2f}% of total Chinese collaborations\n")
    file.write(f"CN - EU collaboration represents {(eu_cn_collaborations / cn_collaborations) * 100:.2f}% of total CN collaborations\n")
    file.write(f"CN - EU collaboration represents {(eu_cn_collaborations / eu_collaborations) * 100:.2f}% of total EU collaborations\n")
    file.write(f"EU - US collaboration represents {(us_eu_collaborations / us_collaborations) * 100:.2f}% of total US collaborations\n")
    file.write(f"EU - US collaboration represents {(us_eu_collaborations / eu_collaborations) * 100:.2f}% of total EU collaborations\n")

with open("computer_science/country_analysis/country_collaboration_cn_us_eu_total.txt", "w") as file:
    file.write(f"{collaboration_type}\n")
    file.write(f"CN - US collaboration {(us_cn_collaborations)}\n")
    file.write(f"CN - EU collaboration {(eu_cn_collaborations)}\n")
    file.write(f"EU - US collaboration {(us_eu_collaborations)}\n")
    file.write(f"EU - US - CN collaboration {(eu_cn_us_collaborations)}\n")


# Define the data
us_data = [us_collaborations, us_eu_collaborations, us_cn_collaborations]
eu_data = [us_eu_collaborations, eu_collaborations, eu_cn_collaborations ]
cn_data = [eu_cn_collaborations, us_cn_collaborations, cn_collaborations]

# Define the x-axis labels
labels = ['US Collaborations', 'EU Collaborations', 'CN Collaborations']

# Define the x-axis locations for each group of bars
x_us = [0, 4, 8]
x_eu = [1, 5, 9]
x_cn = [2, 6, 10]

# Plot the bars
plt.bar(x_us, us_data, color='blue', width=0.8, label='US')
plt.bar(x_eu, eu_data, color='red', width=0.8, label='EU')
plt.bar(x_cn, cn_data, color='green', width=0.8, label='CN')

# Add the x-axis labels and tick marks
plt.xticks([1.5, 5.5, 9.5], labels)
plt.xlabel('Collaboration Type')
plt.ylabel('Number of Collaborations')

# Add a legend
plt.legend()

# Show the plot
plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_cn_us_eu_total.png')
plt.close()

# Define the data
us_data = [(us_collaborations / us_collaborations) * 100, (us_eu_collaborations / us_collaborations) * 100, (us_cn_collaborations / us_collaborations) * 100]
eu_data = [(us_eu_collaborations / eu_collaborations) * 100, (eu_collaborations / eu_collaborations) * 100, (us_cn_collaborations / eu_collaborations) * 100 ]
cn_data = [(us_cn_collaborations / cn_collaborations) * 100, (eu_cn_collaborations / cn_collaborations) * 100, (cn_collaborations / cn_collaborations) * 100]

# Define the x-axis labels
labels = ['US Collaborations', 'EU Collaborations', 'CN Collaborations']

# Define the x-axis locations for each group of bars
x_us = [0, 4, 8]
x_eu = [1, 5, 9]
x_cn = [2, 6, 10]

# Plot the bars
plt.bar(x_us, us_data, color='blue', width=0.8, label='US')
plt.bar(x_eu, eu_data, color='red', width=0.8, label='EU')
plt.bar(x_cn, cn_data, color='green', width=0.8, label='CN')

# Add the x-axis labels and tick marks
plt.xticks([1.5, 5.5, 9.5], labels)
plt.xlabel('Collaboration Type')
plt.ylabel('Number of Collaborations')

# Add a legend
plt.legend()

# Show the plot
plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_cn_us_eu_percent_total.png')
plt.close()