import json

from elasticsearch import Elasticsearch
import sqlite3
import requests

# %%

es = Elasticsearch(["http://localhost:9200"])

# %%

conn = sqlite3.connect("umls.db")
c = conn.cursor()

# %%
# noinspection SqlResolve
cuis = [x[0] for x in c.execute("select distinct(CUI) from MRCONSO;")]
# %%
mrconso = [x[1] for x in c.execute("PRAGMA table_info(MRCONSO);")]
mrdef = [x[1] for x in c.execute("PRAGMA table_info(MRDEF);")]
mrrel = [x[1] for x in c.execute("PRAGMA table_info(MRREL);")]
mrsty = [x[1] for x in c.execute("PRAGMA table_info(MRSTY);")]
columns = ["MRCONSO_" + x for x in mrconso] + ["MRDEF_" + x for x in mrdef]
# %%
req = ""
for i, cui in enumerate(cuis):
    thesaurus = []
    semtypes = []
    relations = []
    # noinspection SqlResolve
    for items in c.execute("select * from MRCONSO conso left join MRDEF def on def.AUI = conso.AUI where conso.CUI = ?;", (cui,)):
        item = dict(zip(columns, items))
        thesaurus.append(item)
    # noinspection SqlResolve
    for items in c.execute("select * from MRREL where CUI1=?;", (cui,)):
        relations.append(dict(zip(mrrel, items)))
    # noinspection SqlResolve
    for items in c.execute("select * from MRSTY where CUI=?;", (cui,)):
        semtypes.append(dict(zip(mrsty, items)))
    doc = {
        "cui": cui,
        "thesaurus": thesaurus,
        "semtypes": semtypes,
        "relations": relations,
    }
    req += '{"index": {"_index": "umls", "_id": "' + cui + '"}}\n'
    req += json.dumps(doc) + "\n"
    if i > 0 and i % 50 == 0:
        resp = requests.post("http://localhost:9200/_bulk", data=req, headers={"Content-Type": "application/x-ndjson"})
        req = ""
        if resp.status_code != 200:
            print(f"*** {resp.status_code}, {resp.text}")
            break
        print("{}/{}".format(i, len(cuis)))
print("done!")
# %%
