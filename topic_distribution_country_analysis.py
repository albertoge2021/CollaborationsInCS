import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from matplotlib import pyplot as plt
from ast import literal_eval
from scipy.stats import ttest_ind

df = pd.read_csv("final.csv")
new_df = pd.DataFrame(
    {
        "country": [],
        "type": [],
        "concepts": [],
    }
)
# df['cities'] = df.cities.apply(lambda x: x[1:-1].split(','))
for index, work in df.iterrows():
    locations = literal_eval(work["location"])
    for location in locations:
        location = dict(location)
        country = location["country"]

        new_df = new_df.append(
            pd.Series(
                [country, work["type"], work["concepts"]],
                index=new_df.columns,
            ),
            ignore_index=True,
        )

new_df = new_df.groupby(["country", "type"]).size().reset_index(name="count")
# new_df = new_df[(new_df["country"] == "GB")]
df_pivot = pd.pivot_table(
    new_df,
    values="count",
    index="country",
    columns="type",
)
ax = df_pivot.plot(kind="bar")
ax.get_figure().set_size_inches(7, 6)
ax.set_xlabel("Countries")
ax.set_ylabel("Count")
print(new_df)
edu = new_df[new_df["type"] == "education"]
com = new_df[new_df["type"] == "company"]
res = ttest_ind(edu["count"], com["count"])
print(res)
plt.show()
