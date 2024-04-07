import os
import csv

# Path to the directory containing TSV files
folder_path = "data"

# Get a list of all files in the directory
files = os.listdir(folder_path)

# Filter files that start with 'cs_' and end with '.tsv'
tsv_files = [file for file in files if file.startswith('cs_works_') and file.endswith('.tsv')]

# Iterate over each TSV file
for file_name in tsv_files:
    file_path = os.path.join(folder_path, file_name)
    print(f"File: {file_name}")
    with open(file_path, 'r', newline='', encoding='utf-8') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        # Count the number of rows
        row_count = sum(1 for _ in tsv_reader)
        print(f"Number of rows: {row_count}")
    print()
