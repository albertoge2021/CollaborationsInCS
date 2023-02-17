from pyalex import Works, Authors, Venues, Institutions, Concepts

#results = Concepts().search_filter(display_name="computer science").get()
works = Institutions().filter(x_concepts ={"id": "C41008148"}).paginate(per_page=200)

for page in works:
    for item in page:
        print(item["geo"])
        
    break