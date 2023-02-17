import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from matplotlib import pyplot as plt
from ast import literal_eval
from scipy.stats import ttest_ind

df = pd.read_csv("output_total.csv")
new_df = pd.DataFrame(
    {"city": [], "state": [], "country": [], "type": [], "concepts": []}
)
# df['cities'] = df.cities.apply(lambda x: x[1:-1].split(','))
for index, work in df.iterrows():
    locations = literal_eval(work["location"])
    for location in locations:
        location = dict(location)
        city = location["city"]
        state = location["state"]
        country = location["country"]
        
        new_df = new_df.append(
            pd.Series(
                [city, state, country, work["type"], work["concepts"]],
                index=new_df.columns,
            ),
            ignore_index=True,
        )
new_df = (
    new_df.groupby(["city", "country", "type"])
    .size()
    .reset_index(name="count")
)
new_df = new_df[(new_df["country"] == "DE") | (new_df["country"] == "AT") | (new_df["country"] == "CH")]
df_pivot = pd.pivot_table(
    new_df,
    values="count",
    index="city",
    columns="type",
)
ax = df_pivot.plot(kind="bar")
ax.get_figure().set_size_inches(7, 6)
ax.set_xlabel("Cities")
ax.set_ylabel("Count")
print(new_df)
edu = new_df[new_df["type"] == "education"]
com = new_df[new_df["type"] == "company"]
res = ttest_ind(edu["count"], com["count"])
print(res)
plt.show()
