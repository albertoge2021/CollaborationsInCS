import pandas as pd
import itertools
import matplotlib.pyplot as plt

# Read the collaboration data into a Pandas DataFrame
df = pd.read_csv("cs_mean.csv")
df = df[df["type"]=="mixed"]
# Extract the countries from the 'location' column
df['continent'] = df['location'].apply(lambda x: x.split(',')[-1].replace("'", "").strip())

# Convert EU country codes to 'EU' string
eu_countries = ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE']

# Calculate the total and mean citations for each country
continent_citations = df.groupby('continent')['citations'].sum().sort_values(ascending=False)
continent_mean_citations = df.groupby('continent')['citations'].mean().sort_values(ascending=False)

# Plot the total and mean citations for each country
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(continent_citations.index, continent_citations.values)
ax1.set_title('Total Citations by continent')
ax1.set_xlabel('continent')
ax1.set_ylabel('Citations')

ax2.bar(continent_mean_citations.index, continent_mean_citations.values)
ax2.set_title('Mean Citations by continent')
ax2.set_xlabel('continent')
ax2.set_ylabel('Citations')

plt.tight_layout()
plt.show()

# Create a list of the countries to compare
countries = ['US', 'China', 'EU']

# Filter the dataframe for the selected countries
selected_countries = df[df['country'].isin(countries)]

# Calculate the mean citations for each selected country
mean_citations = selected_countries.groupby('country')['citations'].mean()

# Plot the mean citations for the selected countries
plt.bar(mean_citations.index, mean_citations.values)
plt.title('Mean Citations by Selected Countries')
plt.xlabel('Country')
plt.ylabel('Citations')
plt.show()