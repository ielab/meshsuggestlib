import xml.etree.ElementTree as ET
from Bio import Entrez
from tqdm import tqdm
import math
import numpy as np
import pandas

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


def temporal_submission(date, query, email):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=10, email=email, mindate=date[0].replace('-', '/'),
                            maxdate=date[1].replace('-', '/'))
    record = Entrez.read(handle)
    count = int(record["Count"])
    return count

def divide_dates(mindate, maxdate, query, email):
    original_chunks = [[mindate, maxdate]]
    final_chunks = []
    while len(original_chunks)>0:
        current_date_range_count = temporal_submission(original_chunks[0], query, email)
        if current_date_range_count > 10000:
            ts1 = pandas.Timestamp(original_chunks[0][0])

            ts2 = pandas.Timestamp(original_chunks[0][1])
            chunk_mean = ts1 + (ts2-ts1)/2
            chunk_mean = str(chunk_mean)[:10]
            chunk_first = [original_chunks[0][0], str(chunk_mean)]
            chunk_last = [str(chunk_mean), original_chunks[0][1]]
            original_chunks.pop(0)
            original_chunks.append(chunk_first)
            original_chunks.append(chunk_last)
        else:
            final_chunks.append(original_chunks[0])
            original_chunks.pop(0)

    return final_chunks

def submit_result(query, email, date_info):
    min_date = date_info[0].replace('/', '-')
    max_date = date_info[1].replace('/', '-')
    date_chunks = divide_dates(min_date, max_date, query, email)
    results = []
    for date_chunk in date_chunks:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=10000, email=email, mindate=date_chunk[0].replace('-', '/'),
                                maxdate=date_chunk[1].replace('-', '/'))
        record = Entrez.read(handle)
        result = record["IdList"]
        results += result
    return results










