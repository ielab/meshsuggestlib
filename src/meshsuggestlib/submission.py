import xml.etree.ElementTree as ET
from Bio import Entrez
from tqdm import tqdm
import math
import numpy as np
import pandas
from pandas import *

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
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=10000, email=email, mindate=date[0].replace('-', '/'),
                                maxdate=date[1].replace('-', '/'))
        record = Entrez.read(handle)
        count = int(record["Count"])
        id_list = record["IdList"]
    except:
        count = 0
        id_list = []
    return count, id_list

def divide_dates(mindate, maxdate, query, email):
    original_chunks = [[mindate, maxdate]]
    id_lists = []
    #final_chunks = []
    while len(original_chunks)>0:
        current_date_range_count, current_id_list = temporal_submission(original_chunks[0], query, email)
        if current_date_range_count > 7000000:
            break
        if current_date_range_count > 10000:
            times = 3
                #math.ceil(current_date_range_count/10000)
            ts1 = pandas.Timestamp(original_chunks[0][0])
            ts2 = pandas.Timestamp(original_chunks[0][1])
            ading_date = (ts2-ts1)/times
            previous_mean = original_chunks[0][0]
            for index in range(1, times):
                chunk_mean = ts1 + (index*ading_date)
                chunk_mean = str(chunk_mean)[:10]
                chunk_now = [previous_mean, str(chunk_mean)]
                previous_mean = str(chunk_mean)
                original_chunks.append(chunk_now)
            chunk_last = [previous_mean, original_chunks[0][1]]
            original_chunks.append(chunk_last)
            original_chunks.pop(0)
        else:
            id_lists.extend(current_id_list)
            original_chunks.pop(0)
            print(original_chunks)
    return set(id_lists)

def submit_result(query, email, date_info):
    min_date = date_info[0].replace('/', '-')
    max_date = date_info[1].replace('/', '-')
    results = divide_dates(min_date, max_date, query, email)
    return results










