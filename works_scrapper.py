import os
import requests.exceptions
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import pandas as pd
from pyalex import Works

# Define a lock for thread safety
lock = threading.Lock()

def get_works(file_path: str, publication_year: int):
    try:
        records = []
        save_threshold = 50000
        query = Works().filter(
            publication_year=publication_year, concept={"id": "C41008148"}
        )
        _, meta = query.get(return_meta=True)
        iterations = 0
        for page in query.paginate(per_page=200, n_max=None):
            iterations += 1
            for record in page:
                countries = []
                institution_types = []
                if (
                    any(
                        "countries" in authorship and not authorship["countries"]
                        for authorship in record["authorships"]
                    )
                    or not record["authorships"]
                ):
                    continue
                for authorship in record["authorships"]:
                    countries.extend(authorship["countries"])
                    for institution in authorship["institutions"]:
                        institution_types.append(
                            institution["type"] if "type" in institution else None
                        )
                if len(countries) < 2:
                    continue
                work = {}
                for key in [
                    "id",
                    "language",
                    "publication_year",
                    "cited_by_count",
                    "type",
                    "is_retracted",
                ]:
                    work[key] = record[key] if key in record else None
                work["institution_types"] = institution_types if institution_types else None
                work["countries"] = countries
                work["num_participants"] = len(record["authorships"])
                work["concepts"] = [
                    concept["display_name"] for concept in record["concepts"]
                ]
                records.append(work)
                if len(records) >= save_threshold:
                    save_records(records, file_path)
                    records = []

        if records:
            save_records(records, file_path)

        print(f"Saved {len(records)} works from {publication_year} to {file_path}.tsv")
    
    except requests.exceptions.ConnectionError:
        # If connection error occurs, delete the file and re-raise the exception
        if os.path.exists(file_path + ".tsv"):
            os.remove(file_path + ".tsv")
        raise

def save_records(records, file_path):
    with lock:
        partial_df = pd.DataFrame.from_records(records)
        partial_df.to_csv(
            file_path + ".tsv",
            sep="\t",
            index=None,
            mode="a",
            header=not Path(file_path + ".tsv").exists(),
        )

def process_year(year):
    file_path = f"computer_science_works/cs_works_{year}"
    get_works(file_path, year)

if __name__ == "__main__":
    Path("computer_science_works/").mkdir(parents=True, exist_ok=True)
    years = [2021]
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_year, years)
