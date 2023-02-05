import os
import time
import logging
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
import pickle
from glob import glob
from torch.utils.data import DataLoader
from meshsuggestlib.arguments import MeSHSuggestLibArguments
from meshsuggestlib.data import EncodeDataset
from meshsuggestlib.retriever import BaseFaissIPRetriever
from meshsuggestlib.evaluation import Evaluator
from meshsuggestlib.data import EncodeCollator
from meshsuggestlib.suggestion import NeuralSuggest, get_mesh_terms, load_mesh_dict, ATM_Suggest, MetaMap_MeSH_Suggestion, UMLS_MeSH_Suggestion
from meshsuggestlib.submission import combine_query, submit_result
from tqdm import tqdm
import torch
from contextlib import nullcontext
import numpy as np
import gc
import sys
logger = logging.getLogger(__name__)

def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup

def write_ranking(topic, final_result, output):
    for rank, r in enumerate(final_result):
        output.write(f'{topic}\t{r}\t{rank + 1}\n')

def read_date(date_file):
    date_dict = {}
    with open(date_file) as f:
        for line in f:
            topic, start_day, end_day = line.split()
            new_start_day = start_day[0:4] + '/' + start_day[4:6] + '/' + start_day[6:8]
            new_end_day = end_day[0:4] + '/' + end_day[4:6] + '/' + end_day[6:8]
            date_dict[topic] = (new_start_day, new_end_day)
    return date_dict

def encoding(dataset, model, tokenizer, max_length, hf_args, mesh_args, encode_is_qry=False):
    encode_loader = DataLoader(
        dataset,
        batch_size=hf_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=hf_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []

    for (batch_ids, batch) in tqdm(encode_loader, desc="Encoding"):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if hf_args.fp16 else nullcontext():
            with torch.no_grad():
                batch.to(mesh_args.device)
                if encode_is_qry:
                    q_reps = model.encode_query(batch)
                    encoded.append(q_reps.cpu())
                else:
                    p_reps = model.encode_passage(batch)
                    encoded.append(p_reps.cpu())

    encoded = torch.cat(encoded)
    return lookup_indices, encoded


def search_queries(retriever, q_reps, p_lookup, args):
    if args.retrieve_batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.retrieve_batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = HfArgumentParser((MeSHSuggestLibArguments, TrainingArguments))
    mesh_args, hf_args = parser.parse_args_into_dataclasses()

    if mesh_args.evaluate_run:
        if os.path.exists(mesh_args.output_file):
            logger.info("Evaluation_parameters input is %s, qrel is %s", mesh_args.output_file, mesh_args.qrel_file)
            if mesh_args.qrel_file != None:
                if os.path.exists(mesh_args.qrel_file):
                    evaluator = Evaluator(mesh_args.qrel_file, ["SetP", "SetR", "SetF", "SetF(beta=3.0)"], mesh_args.output_file)
                    evaluator.compute_metrics()
                    sys.exit()
            else:
                raise Exception("Please put correct qrel file path")
    logger.info("HuggingFace parameters %s", hf_args)
    logger.info("MeSHSuggestLib parameters %s", mesh_args)

    deploy_dataset = mesh_args.dataset
    method = mesh_args.method

    pre_datasets = ['CLEF-2017', 'CLEF-2018', 'CLEF-2019-dta', 'CLEF-2019-intervention']
    pre_methods_base = ['ATM', 'MetaMAP', 'UMLS']
    pre_methods_bert = ['Atomic-BERT', 'Semantic-BERT', 'Fragment-BERT', 'NEW']

    mesh_file = mesh_args.mesh_file

    if deploy_dataset in pre_datasets:
        data_parent_folder = 'data/clef-tar-processed/' + deploy_dataset + '/testing/*'
    else:
        data_parent_folder = 'data/' + deploy_dataset + '/*'
    topics_pathes = glob(data_parent_folder)

    input_keywords_dict = {}
    input_clause_dict = {}

    date_dict = read_date(mesh_args.date_file)
    original_queury_dict = {}
    for topic_path in topics_pathes:
        print(topic_path)
        if not os.path.isdir(topic_path):
            continue
        topic = topic_path.split('/')[-1]
        with open(topic_path+'/original_full_query') as f:
            original_queury_dict[topic] = f.read().strip()
        clause_pathes = glob(topic_path+'/*')
        for clause_path in clause_pathes:
            if not os.path.isdir(clause_path):
                continue
            clause_num = clause_path.split('/')[-1]
            index = topic + '_' + clause_num
            keyword_file = clause_path + '/keywords'
            clause_no_mesh = clause_path + '/clause_no_mesh'

            if not os.path.exists(keyword_file):
                raise Exception("keyword file for topic %s clause num %s does not exist", topic, str(clause_num))
            if not os.path.exists(clause_no_mesh):
                raise Exception("original no mesh clause file for topic %s clause num %s does not exist", topic, str(clause_num))
            current_keywords = []
            with open(clause_no_mesh) as f:
                input_clause_dict[index] = f.read().strip()

            with open(keyword_file, 'r') as f:
                for l in f:
                    k = l.strip()
                    if k not in current_keywords:
                        current_keywords.append(k)
            if len(current_keywords)>0:
                input_keywords_dict[index] = current_keywords
            else:
                raise Exception("no keyword existed for topic %s clause num %s", topic, str(clause_num))

    logger.info("Overall there are %s topic clauses read", str(len(input_keywords_dict)))

    final_query_dict = {}
    if method=="Original":
        try:
            output = open(mesh_args.output_file, 'w')
        except:
            raise Exception("can not create output file at Path %s", mesh_args.output_file)
        for topic_id in original_queury_dict:
            final_query_dict[topic_id] = [original_queury_dict[topic_id]]
    elif method=="Removed":
        try:
            output = open(mesh_args.output_file, 'w')
        except:
            raise Exception("can not create output file at Path %s", mesh_args.output_file)
        for topic_id in input_clause_dict:
            topic_parent = topic_id.split('_')[0]
            no_mesh_c = input_clause_dict[topic_id]
            if topic_parent not in final_query_dict:
                final_query_dict[topic_parent] = []
            new_query = combine_query(no_mesh_c, [])
            final_query_dict[topic_parent].append(new_query)

    elif method in pre_methods_base:
        try:
            output = open(mesh_args.output_file, 'w')
        except:
            raise Exception("can not create output file at Path %s", mesh_args.output_file)
        if deploy_dataset not in pre_datasets:
            if mesh_args.method.lower()=="atm":
                suggester = ATM_Suggest(mesh_args)
            elif mesh_args.method.lower()=="metamap":
                suggester = MetaMap_MeSH_Suggestion()
            elif mesh_args.method.lower() == "umls":
                suggester = UMLS_MeSH_Suggestion()

            for topic_index in tqdm(input_keywords_dict):
                topic_parent = topic_index.split('_')[0]
                input_keywords = input_keywords_dict[topic_index]
                no_mesh_clause = input_clause_dict[topic_index]
                input_dict = {
                    "Keywords": input_keywords,
                    "Type": method,
                }
                result = []
                for a in suggester.suggest(input_dict):
                    result += list(a["MeSH_Terms"].values())
                new_query = combine_query(no_mesh_clause, list(set(result)))
                if topic_parent not in final_query_dict:
                    final_query_dict[topic_parent] = []
                final_query_dict[topic_parent].append(new_query)
        else:
            method_mesh_path = data_parent_folder[:-1] + method + '.res'
            mesh_dict = load_mesh_dict(mesh_args.mesh_file)
            method_mesh_dict = {}
            with open(method_mesh_path) as f:
                for line in f:
                    topic_id, _,mid,_,_,_ = line.split()
                    if topic_id in input_clause_dict:
                        if topic_id not in method_mesh_dict:
                            method_mesh_dict[topic_id] = []
                        method_mesh_dict[topic_id].append(mid)
            for topic_id in input_clause_dict:
                topic_parent = topic_id.split('_')[0]
                no_mesh_c = input_clause_dict[topic_id]
                if topic_parent not in final_query_dict:
                    final_query_dict[topic_parent] = []
                if topic_id in method_mesh_dict:
                    mesh_uids = list(set(method_mesh_dict[topic_id]))
                    mesh_terms_current = get_mesh_terms(mesh_uids, mesh_dict).values()
                    new_query = combine_query(no_mesh_c, mesh_terms_current)
                else:
                    new_query = no_mesh_c
                final_query_dict[topic_parent].append(new_query)
    elif method in pre_methods_bert:
        NeuralPipeline = NeuralSuggest(mesh_args, hf_args)
        if mesh_args.mesh_encoding != None:
            p_reps, p_lookup = pickle_load(mesh_args.mesh_encoding)
        else:
            candidate_dataset = EncodeDataset(mesh_file,
                                              NeuralPipeline.tokenizer,
                                              max_len=mesh_args.p_max_len,
                                              cache_dir=mesh_args.cache_dir)
            print("Dataset tokenized")
            p_lookup, p_reps = encoding(candidate_dataset, NeuralPipeline.model, NeuralPipeline.tokenizer, mesh_args.p_max_len, hf_args, mesh_args)

            with open("model/passage.pt", 'wb') as f:
                pickle.dump((p_reps, p_lookup), f)

            print("Dataset encoded")
        retriever = BaseFaissIPRetriever(p_reps)

        logger.info("Starts Retrieval")
        for topic_index in tqdm(input_keywords_dict):
            input_keywords = input_keywords_dict[topic_index]
            no_mesh_clause = input_clause_dict[topic_index]
            input_dict = {
                "Keywords": input_keywords,
                "Type": method,
            }
            suggestion_results = NeuralPipeline.suggest_mesh_terms(input_dict, retriever, p_lookup)
            suggested_mesh_terms = []
            for r in suggestion_results:
                suggested_mesh_terms += list(r["MeSH_Terms"].values())
            suggested_mesh_terms = list(set(suggested_mesh_terms))
            new_query = combine_query(no_mesh_clause, suggested_mesh_terms)
            topic_parent = topic_index.split('_')[0]
            if topic_parent not in final_query_dict:
                final_query_dict[topic_parent] = []
            final_query_dict[topic_parent].append(new_query)

        try:
            output = open(mesh_args.output_file, 'w')
        except:
            raise Exception("can not create output file at Path %s", mesh_args.output_file)

    for topic_id in final_query_dict:
        mesh_query = " AND ".join(final_query_dict[topic_id])
        print(mesh_query)
        current_d = ('1946/01/01', '2018/12/31')
        if topic_id in date_dict:
            current_d = date_dict[topic_id]
        final_result = submit_result(mesh_query, mesh_args.email, current_d)
        logger.info("topic %s retrieve %s pubmed articles", topic_id, len(final_result))
        write_ranking(topic_id, final_result, output)


if __name__ == "__main__":
    main()
