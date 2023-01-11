import xml.etree.ElementTree as ET
from Bio import Entrez
from tqdm import tqdm
import math

def generateQuery(no_mesh_clause, meshes):
    if len(meshes) > 0:
        mesh_query = "[Mesh] OR ".join(meshes)
        new_query = "(" + mesh_query + "[Mesh] OR " + no_mesh_clause[1:]
    else:
        new_query = no_mesh_clause
    return new_query

def combine_query(no_mesh_clause, mesh_terms):
    new_query = generateQuery(no_mesh_clause, mesh_terms)
    front_num = new_query.count('(')
    back_num = new_query.count(')')
    if front_num != back_num:
        if front_num > back_num:
            new_query = new_query + ')' * (front_num - back_num)
        else:
            new_query = '(' * (back_num - front_num) + new_query
    return new_query

def submit_result(query, email, date_info):
    min_date = date_info[0]
    max_date = date_info[1]
    handle = Entrez.esearch(db="pubmed", term=query, retmax=10, email=email, mindate=min_date,
                            maxdate=max_date)
    record = Entrez.read(handle)
    results = []
    count = int(record["Count"])
    times = math.ceil(count / 100000)
    for index in tqdm(range(0, times)):
        handle = Entrez.esearch(db="pubmed", term=query, retmax=100000, retstart=100000 * index,
                                email=email, mindate=min_date,maxdate=max_date)
        record = Entrez.read(handle)
        result = record["IdList"]
        results += result
    return results









