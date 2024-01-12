# id	title	publication_year	cited_by_count	type	is_retracted	institution_types	countries	concepts

from ast import literal_eval
import pandas as pd
from tqdm import tqdm


file_path = "data_countries/cs_works.csv"
df = pd.read_csv(file_path)
country_type_counts = {}
country_total_counts = {}

# Iterate over each row in the original DataFrame
for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    # Extract information from the current row
    country_list = set(literal_eval(row.countries))
    publication_type = row.type

    # Iterate over each country in the list
    for country in country_list:
        # Update the counts in dictionaries
        key = (country, publication_type)
        country_type_counts[key] = country_type_counts.get(key, 0) + 1
        country_total_counts[country] = country_total_counts.get(country, 0) + 1

# Create a list of dictionaries for each country and type
result_data = []
for (country, publication_type), count in country_type_counts.items():
    percentage = (count / country_total_counts[country]) * 100
    result_data.append(
        {
            "country": country,
            "type": publication_type,
            "total_publications": count,
            "percentage": percentage,
        }
    )

# Create the result DataFrame
result_df = pd.DataFrame(result_data)

result_df = result_df.sort_values(by=["country", "type"]).reset_index(drop=True)

# Save the result DataFrame to a CSV file
result_df.to_csv(
    "paper_results_2/publication_type/publication_type_by_country.csv", index=False
)
