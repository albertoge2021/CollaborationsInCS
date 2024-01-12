# id	title	publication_year	citations	type	is_retracted	institution_types	countries	concepts

from ast import literal_eval
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


df = pd.read_csv("data_countries/cs_works.csv")

# Group by 'is_retracted' and save to file
retracted_grouped = (
    df.groupby("is_retracted").size().reset_index(name="count_retracted")
)
retracted_grouped.to_csv("paper_results_2/retracted/retracted_counts.csv", index=False)

# Group by 'publication_year' and plot
yearly_grouped = (
    df.groupby("publication_year").size().reset_index(name="count_per_year")
)
yearly_grouped.plot(x="publication_year", y="count_per_year", kind="bar", legend=False)
plt.xlabel("Publication Year")
plt.ylabel("Number of Papers")
plt.title("Number of Retracted Papers per Year")
plt.savefig("paper_results_2/retracted/papers_per_year.png")
plt.close()

# Count the number of non-retracted and retracted papers by year
yearly_retracted_count = (
    df.groupby(["publication_year", "is_retracted"]).size().unstack(fill_value=0)
)

# Calculate the percentage of non-retracted and retracted papers by year
yearly_retracted_count["percentage_non_retracted"] = (
    yearly_retracted_count[False]
    / (yearly_retracted_count[False] + yearly_retracted_count[True])
) * 100
yearly_retracted_count["percentage_retracted"] = (
    yearly_retracted_count[True]
    / (yearly_retracted_count[False] + yearly_retracted_count[True])
) * 100

# Save the result to a file
yearly_retracted_count.to_csv(
    "paper_results_2/retracted/yearly_percentage_retracted.csv"
)

# Plot the percentage of retracted papers over the years
plt.figure(figsize=(10, 6))
plt.plot(
    yearly_retracted_count.index,
    yearly_retracted_count["percentage_retracted"],
)
plt.title("Percentage of Retracted Papers Over the Years")
plt.xlabel("Publication Year")
plt.ylabel("Percentage of Retracted Papers")
plt.savefig("paper_results_2/retracted/percentage_retracted_over_years.png")
plt.close()

from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from ast import literal_eval

# Initialize dictionaries with defaultdict
result_by_country = defaultdict(lambda: {"retracted_count": 0, "total_count": 0})
result_by_concept = defaultdict(lambda: {"retracted_count": 0, "total_count": 0})
retracted_by_institution_type = defaultdict(
    lambda: {"retracted_count": 0, "total_count": 0}
)
retracted_by_type = defaultdict(lambda: {"retracted_count": 0, "total_count": 0})
result_by_collaborators = {
    "collaborators": [],
    "retracted_count": [],
    "non_retracted_count": [],
    "percentage_retracted": [],
}
total_by_type = defaultdict(int)

# Iterate through the DataFrame rows with tqdm
for row in tqdm(df.itertuples(), total=len(df), desc="Counting Countries"):
    # Extract information from the row
    country_list = set(literal_eval(row.countries))
    institution_types = (
        set(literal_eval(row.institution_types))
        if pd.notna(row.institution_types)
        else set()
    )
    concepts = set(literal_eval(row.concepts))
    is_retracted = row.is_retracted
    paper_type = row.type
    year = row.publication_year

    # Function to update dictionaries
    def update_dictionary(data, key, is_retracted):
        data[key]["total_count"] += 1
        if is_retracted:
            data[key]["retracted_count"] += 1

    # Update result_by_country dictionary
    for country in country_list:
        update_dictionary(result_by_country, country, is_retracted)

    # Update result_by_concept dictionary
    for concept in concepts:
        update_dictionary(result_by_concept, concept, is_retracted)

    # Update retracted_by_institution_type dictionary
    for paper_type in institution_types:
        update_dictionary(retracted_by_institution_type, paper_type, is_retracted)

    # Update retracted_by_type dictionary
    update_dictionary(retracted_by_type, paper_type, is_retracted)

    # Update total_by_type dictionary
    total_by_type[paper_type] += 1

    # Extract information from the row
    countries = literal_eval(row.countries)
    is_retracted = row.is_retracted

    # Count the number of collaborators
    num_collaborators = len(countries)

    # Update result_by_collaborators dictionary
    if num_collaborators not in result_by_collaborators["collaborators"]:
        result_by_collaborators["collaborators"].append(num_collaborators)
        result_by_collaborators["retracted_count"].append(0)
        result_by_collaborators["non_retracted_count"].append(0)

    index = result_by_collaborators["collaborators"].index(num_collaborators)
    if is_retracted:
        result_by_collaborators["retracted_count"][index] += 1
    else:
        result_by_collaborators["non_retracted_count"][index] += 1

# Convert dictionaries to DataFrames for result_by_country
df_by_country = pd.DataFrame.from_dict(result_by_country, orient="index").reset_index()
df_by_country.columns = ["country", "retracted_count", "total_count"]

# Calculate percentages
df_by_country["percentage_retracted"] = (
    df_by_country["retracted_count"] / df_by_country["total_count"]
) * 100

# Sort DataFrames by highest percentage of retracted papers
df_by_country = df_by_country.sort_values(by="percentage_retracted", ascending=False)

# Save DataFrame to file
df_by_country.to_csv("paper_results_2/retracted/country_results.csv", index=False)


# Convert dictionaries to DataFrames for result_by_concept
df_result_by_concept = pd.DataFrame.from_dict(
    result_by_concept, orient="index"
).reset_index()
df_result_by_concept.columns = ["concept", "retracted_count", "total_count"]

# Calculate percentages
df_result_by_concept["percentage_retracted"] = (
    df_result_by_concept["retracted_count"] / df_result_by_concept["total_count"]
) * 100

# Sort DataFrames by highest percentage of retracted papers
df_result_by_concept = df_result_by_concept.sort_values(
    by="percentage_retracted", ascending=False
)

# Save DataFrame to file
df_result_by_concept.to_csv(
    "paper_results_2/retracted/concepts_percentage.csv", index=False
)


# Convert dictionaries to DataFrames for retracted_by_institution_type
df_retracted_by_institution_type = pd.DataFrame.from_dict(
    retracted_by_institution_type, orient="index"
).reset_index()
df_retracted_by_institution_type.columns = ["type", "retracted_count", "total_count"]

# Calculate percentages
df_retracted_by_institution_type["percentage_retracted"] = (
    df_retracted_by_institution_type["retracted_count"]
    / df_retracted_by_institution_type["total_count"]
) * 100
df_retracted_by_institution_type["percentage_non_retracted"] = (
    100 - df_retracted_by_institution_type["percentage_retracted"]
)

# Sort DataFrames by highest percentage of retracted papers
df_retracted_by_institution_type = df_retracted_by_institution_type.sort_values(
    by="percentage_retracted", ascending=False
)

# Save DataFrame to file
df_retracted_by_institution_type.to_csv(
    "paper_results_2/retracted/retracted_by_institution_type.csv", index=False
)


# Convert dictionaries to DataFrames for retracted_by_type
df_retracted_by_type = pd.DataFrame.from_dict(
    retracted_by_type, orient="index"
).reset_index()
df_retracted_by_type.columns = ["type", "retracted_count", "total_count"]

# Calculate percentages
df_retracted_by_type["percentage_retracted"] = (
    df_retracted_by_type["retracted_count"] / df_retracted_by_type["total_count"]
) * 100

# Sort DataFrames by highest percentage of retracted papers
df_retracted_by_type = df_retracted_by_type.sort_values(
    by="percentage_retracted", ascending=False
)

# Save DataFrame to file
df_retracted_by_type.to_csv(
    "paper_results_2/retracted/retracted_by_type.csv", index=False
)

# Calculate the percentage of retracted papers by collaborators
result_by_collaborators["percentage_retracted"] = [
    (retracted_count / (retracted_count + non_retracted_count)) * 100
    if (retracted_count + non_retracted_count) > 0
    else 0
    for retracted_count, non_retracted_count in zip(
        result_by_collaborators["retracted_count"],
        result_by_collaborators["non_retracted_count"],
    )
]

# Convert dictionary to DataFrame
df_result_by_collaborators = pd.DataFrame(result_by_collaborators)

# Sort DataFrame by collaborators
df_result_by_collaborators = df_result_by_collaborators.sort_values(by="collaborators")

# Save DataFrame to file
df_result_by_collaborators.to_csv(
    "paper_results_2/retracted/collaborators_percentage.csv", index=False
)

# Separate the data into two groups based on is_retracted
retracted_group = df[df["is_retracted"] == True]["citations"]
non_retracted_group = df[df["is_retracted"] == False]["citations"]

from scipy.stats import ttest_ind

# Perform a t-test
t_stat, p_value = ttest_ind(retracted_group, non_retracted_group, equal_var=False)

# Create a DataFrame to store the results
t_test_results = pd.DataFrame(
    {
        "Group": ["Retracted", "Non-Retracted"],
        "Mean": [retracted_group.mean(), non_retracted_group.mean()],
        "Standard Deviation": [retracted_group.std(), non_retracted_group.std()],
        "T-Statistic": [t_stat, None],
        "P-Value": [p_value, None],
    }
)

# Save the DataFrame to a CSV file
t_test_results.to_csv("paper_results_2/retracted/t_test_results.csv", index=False)
