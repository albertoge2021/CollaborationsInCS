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

"""countries = ["EU", "US", "CN"]
df = pd.read_csv("cs_eu.csv")
new_df = pd.DataFrame({"country": [], "collaboration": [], "number": int})
collabs = {country: [] for country in countries}
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
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    country_list= []
    for location in locations:
        country_code = location["country"]
        if country_code in countries:
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
    plt.close()"""

unique_continents = ["CN", "US", "EU"]
for i, continent1 in enumerate(unique_continents):
    for continent2 in unique_continents[i+1:]:
        hm_df_full = pd.read_csv("test_concepts_eu_us_cn.csv")
        test = hm_df_full.groupby("concept")["work"].count().reset_index(name="count").sort_values(by=["count"], ascending=False).head(11)
        test.drop(test[test["concept"] == "Computer science"].index, inplace=True)
        new_df = hm_df_full.loc[hm_df_full["concept"].isin(test.concept.to_list())]
        means_full = new_df.groupby(["year", "concept"]).apply(lambda x: x[x['continent'].isin([continent1, continent2])]['work'].count()).reset_index(name='count')
        means_full.rename(columns={'level_2': 'collaboration'}, inplace=True)
        means_full['collaboration'] = f"{continent1}-{continent2}"
        sns.lineplot(data=means_full, x="year", y="count", hue="concept")
        plt.title(f"Topics per year by concept for {continent1}-{continent2} collaborations")
        plt.savefig(f"computer_science/topic_analysis/line_topics_by_year_{continent1}_{continent2}.png")
        plt.close()

