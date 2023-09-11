import time
import warnings
import asyncio
from urllib3.exceptions import MaxRetryError
from selenium.webdriver.common.by import By
import aiohttp
from diophila import OpenAlex
from geopy.distance import geodesic as GD
from selenium.common.exceptions import WebDriverException, NoSuchElementException
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

openalex = OpenAlex()

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd

CSV_FILE_PATH = "cs_dataset_0.csv"


async def get_work(work, session):
    df = pd.DataFrame(
        {
            "work": str,
            "citations": int,
            "year": int,
            "concepts": [],
            "type": str,
            "countries": [],
            "locations": [],
            "max_distance": float,
            "avg_distance": float,
        }
    )
    if work["doi"] is None:
        return

    citations = work["cited_by_count"] if work["cited_by_count"] else 0
    auth_type_list = []
    country_list = []
    ubication_list = []
    coordinates = []

    for authorship in work["authorships"]:
        for institution in authorship["institutions"]:
            if not "country_code" in institution:
                return
            else:
                country_list.append(institution["country_code"])
            if not institution["type"]:
                return
            else:
                auth_type_list.append(institution["type"])
            if "ror" in institution and institution["ror"]:
                async with session.get(
                    "http://localhost:9292/organizations/"
                    + institution["ror"].rsplit("/", 1)[1]
                ) as resp:
                    req = await resp.json()
                    coordinates.append(
                        (req["addresses"][0]["lat"], req["addresses"][0]["lng"])
                    )
                    ubication_list.append(
                        {
                            "name": req["name"],
                            "type": req["types"][0],
                            "city": req["addresses"][0]["city"],
                            "country": req["country"]["country_code"],
                            "lat": req["addresses"][0]["lat"],
                            "lng": req["addresses"][0]["lng"],
                        }
                    )
            else:
                return
    if len(auth_type_list) < 2:
        return
    if all(item == "company" for item in auth_type_list):
        paper_type = "company"
    elif all(item == "education" for item in auth_type_list):
        paper_type = "education"
    elif all(item in ["education", "company"] for item in auth_type_list):
        paper_type = "mixed"
    else:
        return
    distances = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            coord1 = coordinates[i]
            coord2 = coordinates[j]
            distances.append(GD(coord1, coord2).km)
    avg_distance = sum(distances) / len(distances)
    df = df.append(
        pd.Series(
            [
                work["id"],
                citations,
                work["publication_year"],
                [concept["display_name"] for concept in work["concepts"]],
                paper_type,
                country_list,
                ubication_list,
                max(distances),
                avg_distance,
            ],
            index=df.columns,
        ),
        ignore_index=True,
    )
    df.to_csv(CSV_FILE_PATH, mode="a", index=False, header=False)


async def main():
    # results = Concepts().search_filter(display_name="computer science").get()
    #  "/works?filter=concept.id:C41008148,publication_year:>1989,publication_year:<2022,cited_by_count:<1,authorships.institutions.type:company|education,is_retracted:False,type:journal-article"
    # works_api_url = "https://api.openalex.org/works?filter=concept.id:C41008148,publication_year:>1989,publication_year:<2022,cited_by_count:<13,is_retracted:False,type:journal-article,institutions.country_code:!null"
    # pages_of_works = openalex.get_works_by_api_url(works_api_url)
    header = pd.DataFrame(
        {
            "work": str,
            "citations": int,
            "year": int,
            "concepts": [],
            "type": str,
            "countries": [],
            "locations": [],
            "max_distance": float,
            "avg_distance": float,
        }
    )
    # header.to_csv(CSV_FILE_PATH)
    options = Options()

    driver = webdriver.Chrome(options=options)

    next_cursor = "IlswLCAwLCAnaHR0cHM6Ly9vcGVuYWxleC5vcmcvVzMwMTU5NzgzMzMnXSI="
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False)
    ) as session:
        for page_num in range(1, 225000):
            url = f"https://api.openalex.org/works?filter=concept.id:C41008148,publication_year:%3E1989,publication_year:%3C2022,cited_by_count:0,is_retracted:False,type:journal-article&sort=cited_by_count:desc&per-page=200&cursor={next_cursor}"
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
                    await get_work(result, session)
            else:
                print("error" + str(next_cursor))

            print(str(page_num) + " - " + next_cursor)
            with open("paper_results_2/cursor_0.txt", "a") as file:
                file.write(f"{next_cursor}\n")


asyncio.run(main())
