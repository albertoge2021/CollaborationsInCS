from pyalex import Works, Authors, Venues, Institutions, Concepts
import warnings
import aiohttp
import asyncio

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd

async def get_work(work, session):
    df = pd.DataFrame(
        {"work": [], "year": [], "concepts": [], "type": [], "location": []}
    )
    if work["doi"] == None or all(
        item["institutions"] == [] for item in work["authorships"]
    ):
        return
    auth_type_list = []
    ubication_list = []
    for authorships in work["authorships"]:  # auths
        for institutions in authorships["institutions"]:  # institutions
            auth_type_list.append(
                institutions["type"] if institutions["type"] else None
            )
    if len(auth_type_list) < 2:
        return
    if all(item == "company" for item in auth_type_list):
        paper_type = "company"
        for authorships in work["authorships"]:
            for institutions in authorships["institutions"]:
                if institutions["ror"]:
                    async with session.get(
                        "http://10.0.0.154:9292/organizations/"
                        + institutions["ror"].rsplit("/", 1)[1]
                    ) as resp:
                        req = await resp.json()
                        if req["addresses"][0]["state_code"]:
                            ubication_list.append(
                                {
                                    "name": req["name"],
                                    "type": req["types"][0],
                                    "city": req["addresses"][0]["city"],
                                    "state": req["addresses"][0]["state"],
                                    "country": req["country"]["country_code"],
                                    "lat": req["addresses"][0]["lat"],
                                    "lng": req["addresses"][0]["lng"],
                                }
                            )
                        else:
                            ubication_list.append(
                                {
                                    "name": req["name"],
                                    "type": req["types"][0],
                                    "city": req["addresses"][0]["city"],
                                    "state": req["addresses"][0]["geonames_city"][
                                        "geonames_admin1"
                                    ]["name"],
                                    "country": req["country"]["country_code"],
                                    "lat": req["addresses"][0]["lat"],
                                    "lng": req["addresses"][0]["lng"],
                                }
                            )
    elif all(item == "education" for item in auth_type_list):
        paper_type = "education"
        for authorships in work["authorships"]:
            for institutions in authorships["institutions"]:
                if institutions["ror"]:
                    async with session.get(
                        "http://10.0.0.154:9292/organizations/"
                        + institutions["ror"].rsplit("/", 1)[1]
                    ) as resp:
                        req = await resp.json()
                        if req["addresses"][0]["state_code"]:
                            ubication_list.append(
                                {
                                    "name": req["name"],
                                    "type": req["types"][0],
                                    "city": req["addresses"][0]["city"],
                                    "state": req["addresses"][0]["state"],
                                    "country": req["country"]["country_code"],
                                    "lat": req["addresses"][0]["lat"],
                                    "lng": req["addresses"][0]["lng"],
                                }
                            )
                        else:
                            ubication_list.append(
                                {
                                    "name": req["name"],
                                    "type": req["types"][0],
                                    "city": req["addresses"][0]["city"],
                                    "state": req["addresses"][0]["geonames_city"][
                                        "geonames_admin1"
                                    ]["name"],
                                    "country": req["country"]["country_code"],
                                    "lat": req["addresses"][0]["lat"],
                                    "lng": req["addresses"][0]["lng"],
                                }
                            )
    elif all(item in ["education", "company"] for item in auth_type_list):
        paper_type = "mixed"
        for authorships in work["authorships"]:
            for institutions in authorships["institutions"]:
                if institutions["ror"]:
                    async with session.get(
                        "http://10.0.0.154:9292/organizations/"
                        + institutions["ror"].rsplit("/", 1)[1]
                    ) as resp:
                        req = await resp.json()
                        if req["addresses"][0]["state_code"]:
                            ubication_list.append(
                                {
                                    "name": req["name"],
                                    "type": req["types"][0],
                                    "city": req["addresses"][0]["city"],
                                    "state": req["addresses"][0]["state"],
                                    "country": req["country"]["country_code"],
                                    "lat": req["addresses"][0]["lat"],
                                    "lng": req["addresses"][0]["lng"],
                                }
                            )
                        else:
                            ubication_list.append(
                                {
                                    "name": req["name"],
                                    "type": req["types"][0],
                                    "city": req["addresses"][0]["city"],
                                    "state": req["addresses"][0]["geonames_city"][
                                        "geonames_admin1"
                                    ]["name"],
                                    "country": req["country"]["country_code"],
                                    "lat": req["addresses"][0]["lat"],
                                    "lng": req["addresses"][0]["lng"],
                                }
                            )
    if ubication_list != []:
        df = df.append(
            pd.Series(
                [
                    work["id"],
                    work["publication_year"],
                    [concept["display_name"] for concept in work["concepts"]],
                    paper_type,
                    ubication_list,
                ],
                index=df.columns,
            ),
            ignore_index=True,
        )
    df.to_csv("output.csv", mode="a", index=False, header=False)


async def get_page(page, session):
    for work in page:
        try:
            await get_work(work, session)
        except Exception as e:
            print(e)


async def main():
    # results = Concepts().search_filter(display_name="computer science").get()
    works = (
        Works().filter(concepts={"id": "C41008148"}).paginate(per_page=200, n_max=None)
    )
    header = pd.DataFrame(
        {"work": [], "year": [], "concepts": [], "type": [], "location": []}
    )
    header.to_csv("output.csv")
    pagecon = 0
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False)
    ) as session:
        for page in works:  # pags
            pagecon = pagecon + 1
            print(pagecon)
            await get_page(page, session)

asyncio.run(main())
