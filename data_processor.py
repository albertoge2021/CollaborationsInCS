import pandas as pd

# Read the CSV file
df = pd.read_csv('data/cs_works.tsv', sep='\t')

# Step 1: Remove duplicates based on "id"
initial_rows = len(df)
df = df.drop_duplicates(subset='id')

# Calculate the number of rows removed in the first step
rows_removed_step1 = initial_rows - len(df)
print(f"Step 1: Removed {rows_removed_step1} rows based on duplicates.")

# Step 2: Remove rows where "countries" is an empty list
initial_rows = len(df)
df = df[df['countries'].apply(lambda x: len(eval(x)) > 0)]

# Calculate the number of rows removed in the second step
rows_removed_step2 = initial_rows - len(df)
print(f"Step 2: Removed {rows_removed_step2} rows where 'countries' is an empty list.")

# Save the cleaned DataFrame to a new file
df.to_csv('data/cleaned_cs_works.tsv', sep='\t', index=False)

# Display the final number of rows in the cleaned DataFrame
final_rows = len(df)
print(f"Final DataFrame has {final_rows} rows.")

# Display the total number of rows removed
total_rows_removed = rows_removed_step1 + rows_removed_step2
print(f"Total rows removed: {total_rows_removed}")

"""

Step 1: Removed 734 rows based on duplicates.
Step 2: Removed 32528626 rows where 'countries' is an empty list.
Final DataFrame has 18679541 rows.
Total rows removed: 32529360

"""