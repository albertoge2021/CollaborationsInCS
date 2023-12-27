from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import pandas as pd
from itertools import chain
from pyalex import Works
# Define a lock for thread safety
lock = threading.Lock()


def get_works(file_path: str, publication_year:int):
    records = []
    save_threshold = 20000
    iterations = 0 
    query = Works().filter(publication_year=publication_year, concept={"id": "C41008148"})
    _, meta = query.get(return_meta=True)
    for page in query.paginate(per_page=200, n_max=None):
        for record in page:
            countries = []
            institution_types = []
            for authorship in record["authorships"]:
                if "countries" in authorship:
                    countries.extend(authorship["countries"])
                for institution in authorship["institutions"]:
                    institution_types.append(
                        institution["type"] if "type" in institution else None
                    )
            work = {}
            for key in ['id', 'title', 'publication_year', 'cited_by_count', 'type', 'is_retracted']:
                work[key] = record[key] if key in record else None
            work['institution_types'] = institution_types if institution_types else None
            work['countries'] = countries
            work['concepts'] = [concept["display_name"] for concept in record["concepts"]]
            records.append(work)
            if len(records) >= save_threshold:
                iterations += 1
                print("Saving partial file..." + str(publication_year) + " - " + str(len(records) * iterations) + " / " + str(meta["count"]))
                save_records(records, file_path)
                records = []

    if records:
        save_records(records, file_path)

    # Merge partial files into the final dataset
    merge_partial_files(file_path)

    print(f"Saved {len(records)} works from {publication_year} to {file_path}.tsv")


def save_records(records, file_path):
    with lock:
        partial_df = pd.DataFrame.from_records(records)
        partial_df.to_csv(file_path + "_partial.tsv", sep="\t", index=None, mode='a', header=not Path(file_path + "_partial.tsv").exists())


def merge_partial_files(file_path):
    partial_file_path = file_path + "_partial.tsv"
    final_file_path = file_path + ".tsv"

    with lock:
        if Path(final_file_path).exists():
            # Append data from the partial file to the final file
            partial_df = pd.read_csv(partial_file_path, sep="\t")
            partial_df.to_csv(final_file_path, sep="\t", index=None, mode='a', header=False)
        else:
            # Rename the partial file to the final file
            Path(partial_file_path).rename(final_file_path)

        # Remove the partial file
        Path(partial_file_path).unlink()

def process_year(year):
    file_path = f"data/cs_works_{year}"
    get_works(file_path, year)

if __name__ == "__main__":
    Path("data/").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as executor:
        years = [1993, 1997]
        executor.map(process_year, years)