import warnings
from itertools import groupby
from collections import Counter


warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from matplotlib import pyplot as plt
from ast import literal_eval
from scipy.stats import ttest_ind
from eu_country_codes import COUNTRY_CODES

df = pd.read_csv("final.csv")


def all_same(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


# df['cities'] = df.cities.apply(lambda x: x[1:-1].split(','))
def check_percentage():
    new_df = pd.DataFrame(
        {"work": [], "eu": bool, "national": bool, "regional": bool, "local": bool}
    )
    for index, work in df.iterrows():
        locations = literal_eval(work["location"])
        cities = []
        states = []
        countries = []
        for location in locations:
            location = dict(location)
            cities.append(location["city"])
            states.append(location["state"])
            countries.append(location["country"])

        all_eu = all(item in COUNTRY_CODES for item in countries)
        new_df = new_df.append(
            pd.Series(
                [work, all_eu, all_same(countries), all_same(states), all_same(cities)],
                index=new_df.columns,
            ),
            ignore_index=True,
        )
    print(new_df.eu.value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
    print(
        new_df.national.value_counts(normalize=True).mul(100).round(1).astype(str) + "%"
    )
    print(
        new_df.regional.value_counts(normalize=True).mul(100).round(1).astype(str) + "%"
    )
    print(new_df.local.value_counts(normalize=True).mul(100).round(1).astype(str) + "%")


def check_relationships():
    new_df = pd.DataFrame(
        {"country": [], "collaboration": str, "concepts": [], "type": []}
    )
    # df['cities'] = df.cities.apply(lambda x: x[1:-1].split(','))
    all_countries = []
    for index, work in df.iterrows():
        locations = literal_eval(work["location"])
        for location in locations:
            location = dict(location)
            all_countries.append(location["country"])

    all_countries = list(dict.fromkeys(all_countries))
    for country in all_countries:
        collaborations = []
        concepts = []
        for index, work in df.iterrows():
            locations = literal_eval(work["location"])
            if country in [location["country"] for location in locations]:
                work_collab = []
                concepts.append(work["concepts"])
                for location in locations:
                    work_collab.append(location["country"])
                work_collab.remove(country)
                for item in work_collab:
                    collaborations.append(item)
            else:
                continue
        new_df = new_df.append(
            pd.Series(
                [country, collaborations, concepts, work["type"]],
                index=new_df.columns,
            ),
            ignore_index=True,
        )
    new_df = new_df[(new_df["country"] == "DE")]
    df_pivot = pd.pivot_table(
        new_df,
        values=Counter(list(list(new_df["collaboration"])[0])).values(),
        index=Counter(list(list(new_df["collaboration"])[0])).keys(),
        columns="type",
    )
    ax = df_pivot.plot(kind="bar")
    ax.get_figure().set_size_inches(7, 6)
    ax.set_xlabel("Countries")
    ax.set_ylabel("Count")
    plt.show()


# check_percentage()
check_relationships()
