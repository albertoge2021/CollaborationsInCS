import time
import warnings
import asyncio
from urllib3.exceptions import MaxRetryError
from selenium.webdriver.common.by import By
import aiohttp
from diophila import OpenAlex
from selenium.common.exceptions import WebDriverException, NoSuchElementException
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

openalex = OpenAlex()

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd

eu_countries = [
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
]


async def get_work(work):
    df = pd.DataFrame(
        {
            "work": str,
            "citations": int,
            "year": int,
            "concepts": [],
            "type": str,
            "countries": [],
        }
    )
    if work["doi"] == None:
        return
    citations = 0
    auth_type_list = []
    country_list = []
    for authorship in work["authorships"]:
        for institution in authorship["institutions"]:
            if not "country_code" in institution:
                return
            else:
                country_code = (
                    institution["country_code"]
                    if institution["country_code"] not in eu_countries
                    else "EU"
                )
                country_list.append(country_code)
            if not institution["type"]:
                return
            else:
                auth_type_list.append(institution["type"])
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
    df = df.append(
        pd.Series(
            [
                work["id"],
                citations,
                work["publication_year"],
                [concept["display_name"] for concept in work["concepts"]],
                paper_type,
                country_list,
            ],
            index=df.columns,
        ),
        ignore_index=True,
    )
    df.to_csv("test.csv", mode="a", index=False, header=False)


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
        }
    )
    header.to_csv("test.csv")

    options = Options()
    options.add_argument("--headless")

    # Create a webdriver using the ChromeOptions object
    driver = webdriver.Chrome(options=options)

    # Create a webdriver using the ChromeOptions object
    try:
        driver = webdriver.Chrome(options=options)
    except WebDriverException as e:
        print("Error creating Chrome webdriver:", e)
        exit()

    # Loop through the pages
    next_cursor = "IlswLCAwLCAnaHR0cHM6Ly9vcGVuYWxleC5vcmcvVzMwMQnXSI="
    for page_num in range(1, 215000):
        print(next_cursor)
        # Build the URL for the page
        url = f"https://api.openalex.org/works?filter=concept.id:C41008148,publication_year:%3E1989,publication_year:%3C2022,cited_by_count:0,is_retracted:False,type:journal-article&sort=cited_by_count:desc&per-page=200&cursor={next_cursor}"
        driver.get(url)

        # Get the JSON data from the page source
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

    driver.quit()


asyncio.run(main())
