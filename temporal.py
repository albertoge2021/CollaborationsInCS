import pandas as pd
import os

# Path to the folder containing TSV files
folder_path = "data_countries"

# Get a list of all TSV files in the folder
tsv_files = [file for file in os.listdir(folder_path) if file.endswith(".tsv")]

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through each TSV file and append its data to the combined_data DataFrame
for tsv_file in tsv_files:
    file_path = os.path.join(folder_path, tsv_file)
    data = pd.read_csv(file_path, delimiter="\t")
    combined_data = combined_data.append(data, ignore_index=True)

# Drop duplicates from the combined DataFrame
combined_data = combined_data.drop_duplicates()

# Write the combined and deduplicated data to a new TSV file
output_file_path = "combined_data.tsv"
combined_data.to_csv(output_file_path, sep="\t", index=False)

print(f"Combined data saved to {output_file_path}")
