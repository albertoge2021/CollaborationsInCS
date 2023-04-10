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
df = pd.read_csv("cs_mean.csv")
eu_df = pd.read_csv("cs_eu.csv")
unique_collaboration_types = df["type"].unique()
selected_countries=["US","CN","EU"]
## CONTINENT - COUNTRY ANALYSIS

#region continent
eu_selected_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
Path("computer_science/continent_analysis/").mkdir(parents=True, exist_ok=True)
Path("computer_science/country_analysis/").mkdir(parents=True, exist_ok=True)

df[["type","citations", "international"]].groupby(["type", "international"]).describe().to_csv("computer_science/continent_analysis/describe_citations_by_continent_by_type.csv")
with open('computer_science/continent_analysis/kruskal_citations_by_international_by_type.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Kruskal Test for citations by type" + str(str(stats.kruskal(df[(df['type'] == collaboration_type) & (df['international'] == True)]['citations'],df[(df['type'] == collaboration_type) & (df['international'] == False)]['citations']))))

for collaboration_type in unique_collaboration_types:
    collab_df = df[df["type"] == collaboration_type]
    new_df = pd.DataFrame({"continent": [], "collaboration": [], "number": int})
    collabs = {"NA": [], "EU": [], "AS": [], "OC": [], "SA": [], "AF": []}
    for index, work in collab_df.iterrows():
        continent_list= []
        locations = literal_eval(work["location"])
        for continent in locations:
            continent_list.append(continent["continent"])
        for i in range(len(continent_list)):
            for j in range(i+1, len(continent_list)):
                collabs[continent_list[i]] += [continent_list[j]]
                collabs[continent_list[j]] += [continent_list[i]]
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
    plt.savefig(f'computer_science/continent_analysis/pie_continent_collaboration_by_type_{collaboration_type}.png')
    plt.close()

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
plt.savefig(f'computer_science/continent_analysis/pie_continent_collaboration.png')
plt.close()

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

for row in tqdm(eu_df.itertuples()):
    locations = literal_eval(row.location)
    country_list= []
    for location in locations:
        country_code = location["country"]
        if country_code in selected_countries:
            country_list.append(country_code)
    citations = row.citations
    country_list = set(country_list)
    if "US" in country_list:
        us_collaborations_total +=1
        if "US" in country_list and len(country_list) == 1:
            us_collaborations += 1
            us_citations += citations
            continue
    if "CN" in country_list:
        cn_collaborations_total +=1
        if "CN" in country_list and len(country_list) == 1:
            cn_collaborations += 1
            cn_citations += citations
            continue
    if "EU" in country_list:
        eu_collaborations_total +=1
        if "EU" in country_list and len(country_list) == 1:
            eu_collaborations += 1
            eu_citations += citations
            continue
    if "EU" in country_list and "CN" in country_list and "US" in country_list:
        eu_cn_us_collaborations += 1
        eu_cn_us_citations += citations
    elif "US" in country_list and "CN" in country_list:
        us_cn_collaborations += 1
        us_cn_citations += citations
    elif "US" in country_list and "EU" in country_list:
        us_eu_collaborations += 1
        us_eu_citations += citations
    elif "EU" in country_list and "CN" in country_list:
        eu_cn_collaborations += 1
        eu_cn_citations += citations

with open(f"computer_science/country_analysis/country_collaboration_cn_us_eu_percentage.txt", "w") as file:
    file.write(f"US - US only collaboration represents {(us_collaborations / us_collaborations_total) * 100:.2f}% of total US collaborations\n")
    file.write(f"CN - CN only collaboration represents {(cn_collaborations / cn_collaborations_total) * 100:.2f}% of total CN collaborations\n")
    file.write(f"EU - EU only collaboration represents {(eu_collaborations / eu_collaborations_total) * 100:.2f}% of total EU collaborations\n")
    file.write(f"CN - US collaboration represents {(us_cn_collaborations / us_collaborations_total) * 100:.2f}% of total US collaborations\n")
    file.write(f"CN - US collaboration represents {(us_cn_collaborations / cn_collaborations_total) * 100:.2f}% of total Chinese collaborations\n")
    file.write(f"CN - EU collaboration represents {(eu_cn_collaborations / cn_collaborations_total) * 100:.2f}% of total CN collaborations\n")
    file.write(f"CN - EU collaboration represents {(eu_cn_collaborations / eu_collaborations_total) * 100:.2f}% of total EU collaborations\n")
    file.write(f"EU - US collaboration represents {(us_eu_collaborations / us_collaborations_total) * 100:.2f}% of total US collaborations\n")
    file.write(f"EU - US collaboration represents {(us_eu_collaborations / eu_collaborations_total) * 100:.2f}% of total EU collaborations\n")

with open(f"computer_science/country_analysis/country_collaboration_cn_us_eu.txt", "w") as file:
    file.write(f"US - US only collaboration {(us_collaborations)}\n")
    file.write(f"CN - CN only collaboration {(cn_collaborations)}\n")
    file.write(f"EU - EU only collaboration {(eu_collaborations)}\n")
    file.write(f"CN - US collaboration {(us_cn_collaborations)}\n")
    file.write(f"CN - EU collaboration {(eu_cn_collaborations)}\n")
    file.write(f"EU - US collaboration {(us_eu_collaborations)}\n")
    file.write(f"EU - US - CN collaboration {(eu_cn_us_collaborations)}\n")


# Define the data
us_data = [us_collaborations, us_eu_collaborations, us_cn_collaborations]
eu_data = [us_eu_collaborations, eu_collaborations, eu_cn_collaborations ]
cn_data = [us_cn_collaborations,eu_cn_collaborations, cn_collaborations]

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
plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_cn_us_eu.png')
plt.close()

# Define the data
us_data = [(us_collaborations / us_collaborations_total) * 100, (us_eu_collaborations / us_collaborations_total) * 100, (us_cn_collaborations / us_collaborations_total) * 100]
eu_data = [(us_eu_collaborations / eu_collaborations_total) * 100, (eu_collaborations / eu_collaborations_total) * 100, (us_cn_collaborations / eu_collaborations_total) * 100 ]
cn_data = [(us_cn_collaborations / cn_collaborations_total) * 100, (eu_cn_collaborations / cn_collaborations_total) * 100, (cn_collaborations / cn_collaborations_total) * 100]

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
plt.ylabel('Percentage of Collaborations')

# Add a legend
plt.legend()

# Show the plot
plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_cn_us_eu_percent.png')
plt.close()

us_mean_citations = us_citations / us_collaborations if us_collaborations > 0 else 0
eu_mean_citations = eu_citations / eu_collaborations if eu_collaborations > 0 else 0
cn_mean_citations = cn_citations / cn_collaborations if cn_collaborations > 0 else 0
us_eu_mean_citations = us_eu_citations / us_eu_collaborations if us_eu_collaborations > 0 else 0
us_cn_mean_citations = us_cn_citations / us_cn_collaborations if us_cn_collaborations > 0 else 0
eu_cn_mean_citations = eu_cn_citations / eu_cn_collaborations if eu_cn_collaborations > 0 else 0
eu_cn_us_mean_citations = eu_cn_us_citations / eu_cn_us_collaborations if eu_cn_us_collaborations > 0 else 0

with open("computer_science/country_analysis/country_collaboration_cn_us_eu_citation_mean.txt", "w") as f:
    f.write(f"US mean citations: {us_mean_citations}\n")
    f.write(f"EU mean citations: {eu_mean_citations}\n")
    f.write(f"CN mean citations: {cn_mean_citations}\n")
    f.write(f"US-EU mean citations: {us_eu_mean_citations}\n")
    f.write(f"US-CN mean citations: {us_cn_mean_citations}\n")
    f.write(f"EU-CN mean citations: {eu_cn_mean_citations}\n")
    f.write(f"EU-CN-US mean citations: {eu_cn_us_mean_citations}\n")

# Define the data
us_data_means = [us_mean_citations, us_eu_mean_citations, us_cn_mean_citations]
eu_data_means = [us_eu_mean_citations, eu_mean_citations, eu_cn_mean_citations]
cn_data_means = [us_cn_mean_citations, eu_cn_mean_citations, cn_mean_citations]

# Define the x-axis labels
labels = ['US Collaborations', 'EU Collaborations', 'CN Collaborations']

# Define the x-axis locations for each group of bars
x_us = [0, 4, 8]
x_eu = [1, 5, 9]
x_cn = [2, 6, 10]

# Plot the bars
plt.bar(x_us, us_data_means, color='blue', width=0.8, label='US')
plt.bar(x_eu, eu_data_means, color='red', width=0.8, label='EU')
plt.bar(x_cn, cn_data_means, color='green', width=0.8, label='CN')

# Add the x-axis labels and tick marks
plt.xticks([1.5, 5.5, 9.5], labels)
plt.xlabel('Collaboration Type')
plt.ylabel('Number of Collaborations')

# Add a legend
plt.legend()

# Show the plot
plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_citations_cn_us_eu.png')
plt.close()

#endregion
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
        locations = literal_eval(row.location)
        country_list= []
        for location in locations:
            country_code = location["country"]
            if country_code in selected_countries:
                country_list.append(country_code)
        citations = row.citations
        country_list = set(country_list)
        if "US" in country_list:
            us_collaborations_total +=1
            if "US" in country_list and len(country_list) == 1:
                us_collaborations += 1
                us_citations += citations
                continue
        if "CN" in country_list:
            cn_collaborations_total +=1
            if "CN" in country_list and len(country_list) == 1:
                cn_collaborations += 1
                cn_citations += citations
                continue
        if "EU" in country_list:
            eu_collaborations_total +=1
            if "EU" in country_list and len(country_list) == 1:
                eu_collaborations += 1
                eu_citations += citations
                continue
        if "EU" in country_list and "CN" in country_list and "US" in country_list:
            eu_cn_us_collaborations += 1
            eu_cn_us_citations += citations
        elif "US" in country_list and "CN" in country_list:
            us_cn_collaborations += 1
            us_cn_citations += citations
        elif "US" in country_list and "EU" in country_list:
            us_eu_collaborations += 1
            us_eu_citations += citations
        elif "EU" in country_list and "CN" in country_list:
            eu_cn_collaborations += 1
            eu_cn_citations += citations

    with open(f"computer_science/country_analysis/country_collaboration_cn_us_eu_percentage_type_{collaboration_type}.txt", "w") as file:
        file.write(f"US - US only collaboration represents {(us_collaborations / us_collaborations_total) * 100:.2f}% of total US collaborations\n")
        file.write(f"CN - CN only collaboration represents {(cn_collaborations / cn_collaborations_total) * 100:.2f}% of total CN collaborations\n")
        file.write(f"EU - EU only collaboration represents {(eu_collaborations / eu_collaborations_total) * 100:.2f}% of total EU collaborations\n")
        file.write(f"CN - US collaboration represents {(us_cn_collaborations / us_collaborations_total) * 100:.2f}% of total US collaborations\n")
        file.write(f"CN - US collaboration represents {(us_cn_collaborations / cn_collaborations_total) * 100:.2f}% of total Chinese collaborations\n")
        file.write(f"CN - EU collaboration represents {(eu_cn_collaborations / cn_collaborations_total) * 100:.2f}% of total CN collaborations\n")
        file.write(f"CN - EU collaboration represents {(eu_cn_collaborations / eu_collaborations_total) * 100:.2f}% of total EU collaborations\n")
        file.write(f"EU - US collaboration represents {(us_eu_collaborations / us_collaborations_total) * 100:.2f}% of total US collaborations\n")
        file.write(f"EU - US collaboration represents {(us_eu_collaborations / eu_collaborations_total) * 100:.2f}% of total EU collaborations\n")

    with open(f"computer_science/country_analysis/country_collaboration_cn_us_eu_type_{collaboration_type}.txt", "w") as file:
        file.write(f"US - US only collaboration {(us_collaborations)}\n")
        file.write(f"CN - CN only collaboration {(cn_collaborations)}\n")
        file.write(f"EU - EU only collaboration {(eu_collaborations)}\n")
        file.write(f"CN - US collaboration {(us_cn_collaborations)}\n")
        file.write(f"CN - EU collaboration {(eu_cn_collaborations)}\n")
        file.write(f"EU - US collaboration {(us_eu_collaborations)}\n")
        file.write(f"EU - US - CN collaboration {(eu_cn_us_collaborations)}\n")


    # Define the data
    us_data = [us_collaborations, us_eu_collaborations, us_cn_collaborations]
    eu_data = [us_eu_collaborations, eu_collaborations, eu_cn_collaborations ]
    cn_data = [us_cn_collaborations,eu_cn_collaborations, cn_collaborations]

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
    plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_cn_us_eu_type_{collaboration_type}.png')
    plt.close()

    # Define the data
    us_data = [(us_collaborations / us_collaborations_total) * 100, (us_eu_collaborations / us_collaborations_total) * 100, (us_cn_collaborations / us_collaborations_total) * 100]
    eu_data = [(us_eu_collaborations / eu_collaborations_total) * 100, (eu_collaborations / eu_collaborations_total) * 100, (us_cn_collaborations / eu_collaborations_total) * 100 ]
    cn_data = [(us_cn_collaborations / cn_collaborations_total) * 100, (eu_cn_collaborations / cn_collaborations_total) * 100, (cn_collaborations / cn_collaborations_total) * 100]

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
    plt.ylabel('Percentage of Collaborations')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_cn_us_eu_percent_type_{collaboration_type}.png')
    plt.close()

    us_mean_citations = us_citations / us_collaborations if us_collaborations > 0 else 0
    eu_mean_citations = eu_citations / eu_collaborations if eu_collaborations > 0 else 0
    cn_mean_citations = cn_citations / cn_collaborations if cn_collaborations > 0 else 0
    us_eu_mean_citations = us_eu_citations / us_eu_collaborations if us_eu_collaborations > 0 else 0
    us_cn_mean_citations = us_cn_citations / us_cn_collaborations if us_cn_collaborations > 0 else 0
    eu_cn_mean_citations = eu_cn_citations / eu_cn_collaborations if eu_cn_collaborations > 0 else 0
    eu_cn_us_mean_citations = eu_cn_us_citations / eu_cn_us_collaborations if eu_cn_us_collaborations > 0 else 0

    with open(f"computer_science/country_analysis/country_collaboration_cn_us_eu_citation_mean_type_{collaboration_type}.txt", "w") as f:
        f.write(f"US mean citations: {us_mean_citations}\n")
        f.write(f"EU mean citations: {eu_mean_citations}\n")
        f.write(f"CN mean citations: {cn_mean_citations}\n")
        f.write(f"US-EU mean citations: {us_eu_mean_citations}\n")
        f.write(f"US-CN mean citations: {us_cn_mean_citations}\n")
        f.write(f"EU-CN mean citations: {eu_cn_mean_citations}\n")
        f.write(f"EU-CN-US mean citations: {eu_cn_us_mean_citations}\n")

    # Define the data
    us_data_means = [us_mean_citations, us_eu_mean_citations, us_cn_mean_citations]
    eu_data_means = [us_eu_mean_citations, eu_mean_citations, eu_cn_mean_citations]
    cn_data_means = [us_cn_mean_citations, eu_cn_mean_citations, cn_mean_citations]

    # Define the x-axis labels
    labels = ['US Collaborations', 'EU Collaborations', 'CN Collaborations']

    # Define the x-axis locations for each group of bars
    x_us = [0, 4, 8]
    x_eu = [1, 5, 9]
    x_cn = [2, 6, 10]

    # Plot the bars
    plt.bar(x_us, us_data_means, color='blue', width=0.8, label='US')
    plt.bar(x_eu, eu_data_means, color='red', width=0.8, label='EU')
    plt.bar(x_cn, cn_data_means, color='green', width=0.8, label='CN')

    # Add the x-axis labels and tick marks
    plt.xticks([1.5, 5.5, 9.5], labels)
    plt.xlabel('Collaboration Type')
    plt.ylabel('Number of Collaborations')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(f'computer_science/country_analysis/bar_country_collaboration_citations_cn_us_eu_type_{collaboration_type}.png')
    plt.close()

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
    locations = literal_eval(row.location)
    country_list= []
    for location in locations:
        country_list.append(location["country"])
    if any(country in selected_countries for country in country_list):
        num_countries = len(country_list)
        citations = row.citations
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
            us_ratio_total.append(((us_counts/num_countries), citations, row.type))
        if eu_counts > 0:
            eu_ratio_total.append(((eu_counts/num_countries), citations, row.type))
        if cn_counts > 0:
            cn_ratio_total.append(((cn_counts/num_countries), citations, row.type))

df = pd.DataFrame(us_ratio_total, columns=['ratio', 'citations', 'type'])

with open('computer_science/country_analysis/correlation_ratio_compared_to_citations_by_type_us.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
    f.write("Pearson test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')
    f.write("Spearman test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')

sns.lmplot(
    x="ratio",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('US participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_by_type_us.png')
plt.close()

sns.lmplot(
    x="ratio",
    y="citations",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('US participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_us.png')
plt.close()

means = df.groupby(['ratio',"type"])['citations'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="ratio", y="mean", hue="type")
plt.xlabel('Participation ratio')
plt.ylabel('Mean citations')
plt.title('US participation ratio vs mean citations')
plt.savefig(f'computer_science/country_analysis/scatter_mean_citations_by_ratio_by_type_us.png')
plt.close()

df = pd.DataFrame(eu_ratio_total, columns=['ratio', 'citations', 'type'])

with open('computer_science/country_analysis/correlation_ratio_compared_to_citations_by_type_eu.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
    f.write("Pearson test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')
    f.write("Spearman test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')

sns.lmplot(
    x="ratio",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('EU participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_by_type_eu.png')
plt.close()

sns.lmplot(
    x="ratio",
    y="citations",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('EU participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_eu.png')
plt.close()

means = df.groupby(['ratio', "type"])['citations'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="ratio", y="mean", hue="type")
plt.xlabel('Participation ratio')
plt.ylabel('Mean citations')
plt.title('EU participation ratio vs mean citations')
plt.savefig(f'computer_science/country_analysis/scatter_mean_citations_by_ratio_by_type_eu.png')
plt.close()

df = pd.DataFrame(cn_ratio_total, columns=['ratio', 'citations', 'type'])

with open('computer_science/country_analysis/correlation_ratio_compared_to_citations_by_type_cn.txt', 'w') as f:
    for collaboration_type in unique_collaboration_types:
        f.write("Pearson test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
        f.write("Spearman test for "+collaboration_type +" - " + str(stats.pearsonr(df[df['type'] == collaboration_type]['ratio'], df[df['type'] == collaboration_type]['citations'])))
        f.write('\n')
    f.write("Pearson test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')
    f.write("Spearman test general - " + str(stats.pearsonr(df['ratio'], df['citations'])))
    f.write('\n')

sns.lmplot(
    x="ratio",
    y="citations",
    hue="type",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('CN participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_by_type_cn.png')
plt.close()

sns.lmplot(
    x="ratio",
    y="citations",
    data=df,
    scatter=False,
)
plt.xlabel('Participation ratio') 
plt.ylabel('Number of Citations') 
plt.title('CN participation ratio vs citations')
plt.savefig(f'computer_science/country_analysis/scatter_ratio_citations_cn.png')
plt.close()

means = df.groupby(['ratio',"type"])['citations'].mean().reset_index(name="mean")
sns.lineplot(data=means, x="ratio", y="mean", hue="type")
plt.xlabel('Participation ratio')
plt.ylabel('Mean citations')
plt.title('CN participation ratio vs mean citations')
plt.savefig(f'computer_science/country_analysis/scatter_mean_citations_by_ratio_by_type_cn.png')
plt.close()