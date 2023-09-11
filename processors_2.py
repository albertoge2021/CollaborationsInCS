import pandas as pd
from ast import literal_eval
import pycountry_convert as pc
import warnings
import statistics
from geopy.distance import geodesic as GD
from tqdm import tqdm
from geopy import distance

warnings.simplefilter(action="ignore", category=FutureWarning)

# Merge csvs
df_all = pd.read_csv("cs_dataset.csv")
df_all = df_all.drop_duplicates()
df_all = df_all[df_all["year"] > 1989]
df_all = df_all[df_all["year"] < 2022]
df_0 = pd.read_csv("cs_dataset_0.csv")
df_0 = df_0.drop_duplicates()
df_0 = df_0[df_0["year"] > 1989]
df_0 = df_0[df_0["year"] < 2022]

cs_df = pd.concat([df_all, df_0])
cs_df.reset_index(drop=True)
cs_df.drop_duplicates()

cs_df.to_csv("cs_dataset_final.csv")
