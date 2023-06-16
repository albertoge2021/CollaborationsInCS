import pandas as pd
from ast import literal_eval
import pycountry_convert as pc
import warnings
import statistics
from geopy.distance import geodesic as GD
from tqdm import tqdm
from geopy import distance

warnings.simplefilter(action="ignore", category=FutureWarning)

"""# Merge csvs
df_all = pd.read_csv("cs_all.csv")
df_all = df_all.drop_duplicates()
df_all = df_all[df_all['distance'] > 0]
df_all = df_all[df_all['year'] > 1989]
df_all = df_all[df_all['year'] < 2022]
df_15 = pd.read_csv("cs_15.csv")
df_15 = df_15.drop_duplicates()
df_15 = df_15[df_15['distance'] > 0]
df_15 = df_15[df_15['year'] > 1989]
df_15 = df_15[df_15['year'] < 2022]
df_0 = pd.read_csv("cs_0.csv")
df_0 = df_0.drop_duplicates()
df_0 = df_0[df_0['distance'] > 0]
df_0 = df_0[df_0['year'] > 1989]
df_0 = df_0[df_0['year'] < 2022]

cs_df = pd.concat([df_all, df_15,df_0])
cs_df.reset_index(drop=True)
cs_df.drop_duplicates()

cs_df.to_csv("cs.csv")

# Setup Data with continent and scope
df = pd.read_csv("cs.csv")
dev_df = pd.read_csv("human_dev_standard.csv")
df['no_dev'] = False
df['international'] = False
dev_countries = ['AO', 'BJ', 'BF', 'BI', 'CF', 'TD', 'KM', 'CD', 'DJ', 'ER', 'ET', 'GM', 'GN', 'GW', 'LS', 'LR', 'MG', 'MW', 'ML', 'MR', 'MZ', 'NE', 'RW', 'ST', 'SN', 'SL', 'SO', 'SS', 'SD', 'TG', 'UG', 'TZ', 'ZM', 'AF', 'BD', 'BT', 'KH', 'LA', 'MM', 'NP', 'TL', 'YE', 'HT', 'KI', 'SB', 'TV']

for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    new_loc = []
    location_list = []
    education = 0
    company = 0
    first_country = locations[0]["country"]
    for country in locations[1:]:
        if country["country"] != first_country:
            df._set_value(row.Index,'international', True)
    for location in locations:
        location = dict(location)
        try:
            if location["country"] == "TW" or location["country"] == "MO":
                dev_index = list(dev_df.loc[(dev_df['Year']== row.year) & (dev_df["Code_2"] == "CN")]["Index"])[0]
            elif location["country"] == "FO":
                dev_index = list(dev_df.loc[(dev_df['Year']== row.year) & (dev_df["Code_2"] == "DK")]["Index"])[0]
            elif location["country"] == "VG":
                dev_index = list(dev_df.loc[(dev_df['Year']== row.year) & (dev_df["Code_2"] == "GB")]["Index"])[0]                
            else:
                dev_index = list(dev_df.loc[(dev_df['Year']== row.year) & (dev_df["Code_2"] == location["country"])]["Index"])[0]
        except:
            dev_index = None
        location_list.append(dev_index)
        if location["country"] in dev_countries:
            df._set_value(row.Index,'no_dev', True)
        location["dev_index"] = dev_index
        location["continent"] = pc.convert_country_alpha2_to_continent_code.country_alpha2_to_continent_code(location["country"])
        new_loc.append(location)
        if location["type"] == "Company":
            company += 1
        elif location["type"] == "Education" :
            education += 1
    try:
        df._set_value(row.Index,'mean_index', statistics.mean(location_list))
    except:
        df._set_value(row.Index,'mean_index', None)
    df._set_value(row.Index,'ratio', company/(education+company))
    df._set_value(row.Index,'location',new_loc)
df.reset_index(drop=True)
df.to_csv("cs_data.csv")

df = pd.read_csv("cs_data.csv")
df = df.drop_duplicates("work")
df.to_csv("cs.csv")
"""
"""df = pd.read_csv("cs.csv")

for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    distances = []
    for country in locations:
        country_coord = (country["lat"], country["lng"])
        distances.append(country_coord)
    total_distance = 0
    count = 0
    for i in range(len(distances)):
        for j in range(i+1, len(distances)):
            dist = GD(distances[i], distances[j]).km
            total_distance += dist
            count += 1
    mean_distance = total_distance / count
    df._set_value(row.Index,'mean_distance',mean_distance)

df = df[df['mean_index'].notna()]
df.to_csv("cs_mean.csv")

df = pd.read_csv("cs_mean.csv")
df['citations'] = df['citations'].fillna(0)
df.to_csv("cs_mean.csv")"""

"""eu_countries = ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE']
df = pd.read_csv("cs_mean.csv")
for row in tqdm(df.itertuples()):
    locations = literal_eval(row.location)
    new_loc = []
    location_list = []
    for location in locations:
        location = dict(location)
        if location["country"] in eu_countries:
            location["country"] = "EU"
        new_loc.append(location)
    df._set_value(row.Index,'location',new_loc)
df.to_csv("cs_eu.csv")"""

df_all = pd.read_csv("test_df.csv", low_memory=False)

# Create an empty list to store the valid rows
valid_rows = []

# Iterate over the rows of the DataFrame
for row in tqdm(df_all.itertuples()):
    try:
        # Try to convert the year to an integer
        year = int(row.year)
        if year > 1989 and year <= 2022:
            # If the conversion succeeds, append the row to the list
            valid_rows.append(row[1:])
    except ValueError:
        # If the conversion fails, ignore the row and move on to the next one
        print("no year")
        continue

# Convert the list of valid rows to a new DataFrame
df_valid = pd.DataFrame(valid_rows, columns=df_all.columns)

df_all = df_valid.drop_duplicates()
df_all = df_valid.dropna()
# Replace all occurrences of 'GB' with 'EU'
df_all = df_all.replace("GB", "EU")

df_all.to_csv("cs_final_eu_changed.csv", index=False)
