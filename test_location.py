import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
from ast import literal_eval
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

df = pd.read_csv("final.csv")
new_df = pd.DataFrame(
    {
        "city": [],
        "lat": [],
        "lan": [],
        "state": [],
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
        city = location["city"]
        state = location["state"]
        country = location["country"]
        lat = location["lat"]
        lan = location["lng"]
        for concept in list(dict.fromkeys(literal_eval(work["concepts"]))):
            new_df = new_df.append(
                pd.Series(
                    [city, lat, lan, state, country, work["type"], concept],
                    index=new_df.columns,
                ),
                ignore_index=True,
            )
new_df = (
    new_df.groupby(["city", "lat", "lan", "country", "type", "concepts"])
    .size()
    .reset_index(name="count")
)
fig = px.scatter_mapbox(
    new_df,
    lat="lat",
    lon="lan",
    color="type",
    size="count",
    size_max=30,
    category_orders={"type": list(new_df.type.unique())},
    color_discrete_sequence=["#2b83ba", "#fdae61", "#38BA2B"],
    zoom=3,
    mapbox_style="open-street-map",
)
fig.show()
