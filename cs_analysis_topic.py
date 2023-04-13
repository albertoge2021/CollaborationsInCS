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
#dev_df = pd.read_csv("human_dev_standard.csv")
#df = pd.read_csv("cs_mean.csv")
eu_df = pd.read_csv("cs_eu.csv")
unique_collaboration_types = eu_df["type"].unique()
selected_countries = ["US", "CN", "EU"]
"""
## TOPIC ANALYSIS

Path("computer_science/topic_analysis/").mkdir(parents=True, exist_ok=True)
hm_df = pd.DataFrame(
    {
        "work": str,
        "continent": [],
        "concept": [],
        "year":int,
        "no_dev":bool,
    }
)
continent_concept_list = []
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    continents = []
    for continent in locations:
        continents.append(continent["continent"])
    for continent in set(continents):
        continent = "NAA" if continent == "NA" else continent
        concepts = literal_eval(row.concepts)
        for concept in concepts:
            continent_concept_list.append([row.work, continent, concept, row.year, row.no_dev, row.type])
hm_df = pd.DataFrame(continent_concept_list, columns = ['work','continent', 'concept', 'year', 'no_dev', 'type'])
hm_df.to_csv("test_concepts.csv")
hm_df_full = pd.read_csv("test_concepts.csv")
unique_dev_types = df["no_dev"].unique()

for collaboration_type in unique_collaboration_types:
    for dev_type in unique_dev_types:
        if dev_type == True:
            dev_type_name = "no_developed"
        else:
            dev_type_name = "developed"
        test = (
            hm_df_full.groupby("concept")["work"]
            .count()
            .reset_index(name="count")
            .sort_values(by=["count"], ascending=False)
            .head(11)
        )
        test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
        new_df = hm_df_full.loc[hm_df_full["concept"].isin(test.concept.to_list())]
        means = (
            new_df.groupby(["no_dev", "concept", "year", "type"])["work"]
            .count()
            .reset_index(name="count")
        )
        means = means[
            (means["no_dev"] == dev_type) & (means["type"] == collaboration_type)
        ]
        sns.lineplot(data=means, x="year", y="count", hue="concept")
        plt.savefig(
            f"computer_science/topic_analysis/line_topics_by_year_by_development_{dev_type_name}.png"
        )
        plt.close()


# for collaboration_type in unique_collaboration_types:
# for developement_type in unique_dev_types:
unique_continents = ["NAA", "OC", "EU", "AS", "AF", "SA"]
for unique_continent in unique_continents:
    hm_df_full = pd.read_csv("test_concepts.csv")
    test = (
        hm_df_full.groupby("concept")["work"]
        .count()
        .reset_index(name="count")
        .sort_values(by=["count"], ascending=False)
        .head(11)
    )
    test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
    new_df = hm_df_full.loc[hm_df_full["concept"].isin(test.concept.to_list())]
    means_full = (
        new_df.groupby(["continent", "concept", "year"])["work"]
        .count()
        .reset_index(name="count")
    )
    means = means_full[
        (means_full["continent"] == unique_continent)
    ]  # & (means["type"]=="mixed")
    sns.lineplot(data=means, x="year", y="count", hue="concept")
    plt.savefig(
        f"computer_science/topic_analysis/line_topics_by_year_by_contient_{unique_continent}.png"
    )
    plt.close()


new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
collabs = {country: [] for country in selected_countries}
continent_concept_list = []
hm_df = pd.DataFrame(
    {
        "work": str,
        "country": [],
        "concept": [],
        "year":int,
        "no_dev":bool,
    }
)
for row in tqdm(eu_df.itertuples()):
    locations = literal_eval(row.location)
    country_list= []
    for location in locations:
        country_code = location["country"]
        if country_code in selected_countries:
            country_list.append(country_code)
    concepts = literal_eval(row.concepts)
    for concept in concepts:
        for country in country_list:
            continent_concept_list.append([row.work, country, concept, row.year, row.no_dev, row.type])
hm_df = pd.DataFrame(continent_concept_list, columns = ['work','country', 'concept', 'year', 'no_dev', 'type'])
hm_df.to_csv("test_concepts_eu_us_cn.csv")


unique_continents = ["CN", "US", "EU"]
for unique_continent in unique_continents:
    hm_df_full = pd.read_csv("test_concepts_eu_us_cn.csv")
    test = (
        hm_df_full.groupby("concept")["work"]
        .count()
        .reset_index(name="count")
        .sort_values(by=["count"], ascending=False)
        .head(11)
    )
    test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
    new_df = hm_df_full.loc[hm_df_full["concept"].isin(test.concept.to_list())]
    means_full = (
        new_df.groupby(["continent", "concept", "year"])["work"]
        .count()
        .reset_index(name="count")
    )
    means = means_full[
        (means_full["continent"] == unique_continent)
    ]  # & (means["type"]=="mixed")
    sns.lineplot(data=means, x="year", y="count", hue="concept")
    plt.savefig(
        f"computer_science/topic_analysis/line_topics_by_year_by_country_{unique_continent}.png"
    )
    plt.close()

for i, continent1 in enumerate(unique_continents):
    for continent2 in unique_continents[i + 1 :]:
        hm_df_full = pd.read_csv("test_concepts_eu_us_cn.csv")
        test = (
            hm_df_full.groupby("concept")["work"]
            .count()
            .reset_index(name="count")
            .sort_values(by=["count"], ascending=False)
            .head(11)
        )
        test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
        new_df = hm_df_full.loc[hm_df_full["concept"].isin(test.concept.to_list())]
        means_full = (
            new_df.groupby(["year", "concept"])
            .apply(
                lambda x: x[x["continent"].isin([continent1, continent2])][
                    "work"
                ].count()
            )
            .reset_index(name="count")
        )
        means_full.rename(columns={"level_2": "collaboration"}, inplace=True)
        means_full["collaboration"] = f"{continent1}-{continent2}"
        sns.lineplot(data=means_full, x="year", y="count", hue="concept")
        plt.title(
            f"Topics per year by concept for {continent1}-{continent2} collaborations"
        )
        plt.savefig(
            f"computer_science/topic_analysis/line_topics_by_year_{continent1}_{continent2}.png"
        )
        plt.close()"""

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
        if "US" in country_list and "CN" not in country_list and "EU" not in country_list:
            for concept in concepts:
                us_topics.append((concept, row.year, row.type))
            continue

df = pd.DataFrame(us_topics, columns=["concept", "year", "type"])
test = (
    df.groupby('concept').size().reset_index(name='count')
    .sort_values(by=["count"], ascending=False)
    .head(11)
)
test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
new_df = df.loc[df["concept"].isin(test.concept.to_list())]
means_full = (
    new_df.groupby(["concept", "year"]).size().reset_index(name='count')
)
sns.lineplot(data=means_full, x="year", y="count", hue="concept")
plt.savefig(
    f"computer_science/topic_analysis/line_topics_by_year_us.png"
)
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
        if "CN" in country_list and "US" not in country_list and "EU" not in country_list:
            for concept in concepts:
                cn_topics.append((concept, row.year, row.type))
            continue
        else:
            if "US" in country_list and "CN" in country_list and not "EU" in country_list:
                for concept in concepts:
                    cn_us_topics.append((concept, row.year, row.type))
            if "EU" in country_list and "CN" in country_list and not "US" in country_list:
                for concept in concepts:
                    eu_cn_topics.append((concept, row.year, row.type))
    if "EU" in country_list and "CN" in country_list and "US" in country_list:
        for concept in concepts:
                cn_eu_us_topics.append((concept, row.year, row.type))
        continue



df = pd.DataFrame(cn_topics, columns=["concept", "year", "type"])

test = (
    df.groupby('concept').size().reset_index(name='count')
    .sort_values(by=["count"], ascending=False)
    .head(11)
)
test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
new_df = df.loc[df["concept"].isin(test.concept.to_list())]
means_full = (
    new_df.groupby(["concept", "year"]).size().reset_index(name='count')
)
sns.lineplot(data=means_full, x="year", y="count", hue="concept")
plt.savefig(
    f"computer_science/topic_analysis/line_topics_by_year_cn.png"
)
plt.close()

df = pd.DataFrame(cn_eu_us_topics, columns=["concept", "year", "type"])

test = (
    df.groupby('concept').size().reset_index(name='count')
    .sort_values(by=["count"], ascending=False)
    .head(11)
)
test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
new_df = df.loc[df["concept"].isin(test.concept.to_list())]
means_full = (
    new_df.groupby(["concept", "year"]).size().reset_index(name='count')
)
sns.lineplot(data=means_full, x="year", y="count", hue="concept")
plt.savefig(
    f"computer_science/topic_analysis/line_topics_by_year_cn_us_eu.png"
)
plt.close()

df = pd.DataFrame(cn_us_topics, columns=["concept", "year", "type"])

test = (
    df.groupby('concept').size().reset_index(name='count')
    .sort_values(by=["count"], ascending=False)
    .head(11)
)
test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
new_df = df.loc[df["concept"].isin(test.concept.to_list())]
means_full = (
    new_df.groupby(["concept", "year"]).size().reset_index(name='count')
)
sns.lineplot(data=means_full, x="year", y="count", hue="concept")
plt.savefig(
    f"computer_science/topic_analysis/line_topics_by_year_cn_us.png"
)
plt.close()

df = pd.DataFrame(eu_cn_topics, columns=["concept", "year", "type"])

test = (
    df.groupby('concept').size().reset_index(name='count')
    .sort_values(by=["count"], ascending=False)
    .head(11)
)
test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
new_df = df.loc[df["concept"].isin(test.concept.to_list())]
means_full = (
    new_df.groupby(["concept", "year"]).size().reset_index(name='count')
)
sns.lineplot(data=means_full, x="year", y="count", hue="concept")
plt.savefig(
    f"computer_science/topic_analysis/line_topics_by_year_cn_eu.png"
)
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
        if "EU" in country_list and "US" not in country_list and "CN" not in country_list:
            for concept in concepts:
                eu_topics.append((concept, row.year, row.type))
            continue
        else:
            if "US" in country_list and "EU" in country_list and not "CN" in country_list:
                for concept in concepts:
                    us_eu_topics.append((concept, row.year, row.type))

df = pd.DataFrame(eu_topics, columns=["concept", "year", "type"])

test = (
    df.groupby('concept').size().reset_index(name='count')
    .sort_values(by=["count"], ascending=False)
    .head(11)
)
test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
new_df = df.loc[df["concept"].isin(test.concept.to_list())]
means_full = (
    new_df.groupby(["concept", "year"]).size().reset_index(name='count')
)
sns.lineplot(data=means_full, x="year", y="count", hue="concept")
plt.savefig(
    f"computer_science/topic_analysis/line_topics_by_year_eu.png"
)
plt.close()


df = pd.DataFrame(us_eu_topics, columns=["concept", "year", "type"])
test = (
    df.groupby('concept').size().reset_index(name='count')
    .sort_values(by=["count"], ascending=False)
    .head(11)
)
test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
new_df = df.loc[df["concept"].isin(test.concept.to_list())]
means_full = (
    new_df.groupby(["concept", "year"]).size().reset_index(name='count')
)
sns.lineplot(data=means_full, x="year", y="count", hue="concept")
plt.savefig(
    f"computer_science/topic_analysis/line_topics_by_year_us_eu.png"
)
plt.close()