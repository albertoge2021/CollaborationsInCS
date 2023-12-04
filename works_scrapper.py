import asyncio
import warnings
from selenium.webdriver.common.by import By
import aiohttp
from selenium.common.exceptions import NoSuchElementException
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd

CSV_FILE_PATH = "cs_works_dataset.csv"


# This code takes in a work, and extracts a number of features from it
# including the title, year, language, number of citations, type, whether
# it has been retracted, institution type, countries, and concepts.
# These features are then appended to a CSV file.


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
            "countries": list,
            "concepts": list,
        }
    )

    countries = []
    institution_types = []

    for authorship in work["authorships"]:
        for institution in authorship["institutions"]:
            countries.append(
                institution["country"] if "country" in institution else None
            )
            institution_types.append(
                institution["type"] if "type" in institution else None
            )

    df = df.append(
        pd.Series(
            [
                work["id"],
                work["title"],
                work["publication_year"],
                work["language"] if "landuage" in work else None,
                work["cited_by_count"] if "cited_by_count" in work else None,
                work["type"] if "type" in work else None,
                work["retracted"],
                institution_types if institution_types else None,
                countries if countries else None,
                [concept["display_name"] for concept in work["concepts"]],
            ],
            index=df.columns,
        ),
        ignore_index=True,
    )
    df.to_csv(CSV_FILE_PATH, mode="a", index=False, header=False)


# The code takes the JSON data from the API and parses through it to extract key information such as the title, year, language, number of citations, and whether the paper was retracted. It then saves this information to a CSV file and stores the data in a dataframe. It also saves the cursor value to a text file so that the code can continue from where it left off if there is an error or the code needs to be stopped. The code also checks for errors in the JSON data and prints a message if there is an error. If there are no errors, it moves on to the next paper.


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
            "countries": list,
            "concepts": list,
        },
        index=[0],
    )
    header.to_csv(CSV_FILE_PATH, index=False)
    options = Options()

    driver = webdriver.Chrome(options=options)

    next_cursor = "*"
    with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        while next_cursor:
            # Change URL if local version is available
            url = f"https://api.openalex.org/works?filter=concept.id:C41008148,publication_year:%3E1989,publication_year:%3C2023&sort=cited_by_count:desc&per-page=200&cursor={next_cursor}"
            driver.get(url)
            try:
                content = driver.find_element(By.TAG_NAME, "pre").text
            except NoSuchElementException:
                print("Error finding JSON data on page:", url)
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
