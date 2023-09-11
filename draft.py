from ast import literal_eval
from collections import Counter
from matplotlib import patches
import pandas as pd
import warnings
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)


# https://www.cwauthors.com/article/Four-common-statistical-test-ideas-to-share-with-your-academic-colleagues
# https://towardsdatascience.com/what-are-the-commonly-used-statistical-tests-in-data-science-a95cfc2e6b5e

dataset1 = pd.read_csv("cs_dataset_location.csv", header=None)
dataset2 = pd.read_csv("cs_dataset_location_0.csv", header=None)
dataset3 = pd.read_csv("cs_dataset_location_10.csv", header=None)

# Combine the datasets
combined_data = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)

# Remove duplicates
combined_data = combined_data.drop_duplicates()

# Output file name
output_file = "cs_dataset.csv"

# Export the combined data to CSV
combined_data.to_csv(output_file, index=False, header=False)
"""participation_df = pd.DataFrame(collaborators, columns=["origin", "collaborator", "citations", "type"])

# Remove rows with specific collaborators
collaborators_to_remove = ['EU', 'US', 'CN']
participation_df = participation_df[~participation_df['collaborator'].isin(collaborators_to_remove)]

# Group by 'origin' and 'collaborator', and count the occurrences
grouped_data = participation_df.groupby(['origin', 'collaborator']).size().reset_index(name='count')

# Sort the DataFrame by 'count' in descending order
sorted_data = grouped_data.sort_values('count', ascending=False)

# Select the top 15 most common collaborators for each origin
top_15_collaborators = sorted_data.groupby('origin').head(15)

# Plot the bar plot
plt.figure(figsize=(12, 8))
sns.barplot(data=top_15_collaborators, x='collaborator', y='count', hue='origin', palette='Set3')
plt.title('Top 15 Most Common Collaborators by Origin')
plt.xlabel('Collaborator')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Origin')
plt.tight_layout()

# Show the plot
plt.show()"""
