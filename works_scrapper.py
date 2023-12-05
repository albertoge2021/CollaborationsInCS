import asyncio
import warnings
import aiohttp
import json
from urllib.request import urlopen
from pathlib import Path


warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd

CSV_FILE_PATH = "cs_works_dataset.csv"


async def get_work(work):
    df = pd.DataFrame(
        {
            "work": str,
            "title": str,
            "year": int,
            "language": str,
            "citations": int,
            "type": str,
            "retracted": bool,
            "institution_type": str,
            "countries": [],
            "concepts": [],
        }
    )

    countries = []
    institution_types = []

    for authorship in work["authorships"]:
        if "countries" in authorship:
            countries.extend(authorship["countries"])
        for institution in authorship["institutions"]:
            institution_types.append(
                institution["type"] if "type" in institution else None
            )

    if len(countries) <= 1:
        return

    df = df.append(
        pd.Series(
            [
                work["id"],
                work["title"],
                work["publication_year"],
                work["language"] if "language" in work else None,
                work["cited_by_count"] if "cited_by_count" in work else None,
                work["type"] if "type" in work else None,
                work["is_retracted"],
                institution_types if institution_types else None,
                countries,
                [concept["display_name"] for concept in work["concepts"]],
            ],
            index=df.columns,
        ),
        ignore_index=True,
    )
    df.to_csv(CSV_FILE_PATH, mode="a", index=False, header=False)


async def main():
    # Dataframe definition
    header = pd.DataFrame(
        {
            "work": str,
            "title": str,
            "year": int,
            "language": str,
            "citations": int,
            "type": str,
            "retracted": bool,
            "institution_type": str,
            "countries": [],
            "concepts": [],
        }
    )
    csv_path = Path(CSV_FILE_PATH)
    if not csv_path.is_file():
        header.to_csv(CSV_FILE_PATH, index=False)

    next_cursor = "*"
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False)
    ) as session:
        while next_cursor:
            # Change URL if local version is available
            url = f"https://api.openalex.org/works?filter=concept.id:C41008148,institutions_distinct_count:%3E1,publication_year:%3E1989,publication_year:%3C2023&sort=cited_by_count:desc&per-page=200&cursor={next_cursor}"
            try:
                with urlopen(url) as response:
                    content = response.read().decode("utf-8")
            except Exception as e:
                print(f"Error downloading content from {url}: {e}")
                continue
            try:
                parsed_json = json.loads(content)
            except:
                print("Error parsing JSON data from page:", url)
                continue
            next_cursor = parsed_json["meta"]["next_cursor"]
            if "results" in parsed_json:
                for result in parsed_json["results"]:
                    await get_work(result)
            else:
                print("error" + str(next_cursor))

            with open("paper_results_2/cursor.txt", "w") as file:
                file.write(f"{next_cursor}\n")


asyncio.run(main())
