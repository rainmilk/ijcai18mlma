from SPARQLWrapper import SPARQLWrapper, JSON
from pandas import DataFrame
import json
import pandas as pd

movielens_tsv = 'MappingMovielens2DBpedia-1.2.tsv'

s = SPARQLWrapper('http://dbpedia.org/sparql')
s.setReturnFormat(JSON)
ablist = []
castlist = []

data = pd.read_table(movielens_tsv)
maxrow = data.shape[0]
print("Rows: %d"%maxrow)

dataset = []
for i, d in enumerate(data.itertuples(), 1):
    q = """PREFIX movie:<%s>
    select ?abstract ?director ?writer ?starring
     { movie: dbo:abstract ?abstract.
       optional { movie: dbo:director ?director }
       optional { movie: dbo:writer ?writer }
       optional { movie: dbo:starring ?starring }
      FILTER (langMatches(lang(?abstract),"en")) }"""%(d.DBpedia_uri)

    s.setQuery(q)
    results = s.queryAndConvert()
    result = results["results"]["bindings"]
    cast = set()
    if result:
        abs = result[0]["abstract"]["value"]
        for r in result:
            if "director" in r:
                cast.add(r["director"]["value"])
            if "writer" in r:
                cast.add(r["writer"]["value"])
            if "starring" in r:
                cast.add(r["starring"]["value"])
    else:
        abs = 'NA'
    dataset.append({'item_id':str(d.item_id), 'abstract':abs, 'cast':list(cast)})

    print("Progress %d/%d"%(i, maxrow))

dataset_file = "movielens_abstract_cast.json"
with open(dataset_file, 'w') as jsonfile:
    json.dump({'data':dataset}, jsonfile)





