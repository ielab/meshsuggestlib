import os
import time
import logging
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from glob import glob
from meshsuggestlib.callbacks import get_reporting_integration_callbacks, CallbackHandler
from torch.utils.data import DataLoader
from meshsuggestlib.arguments import MeSHSuggestLibArguments
from meshsuggestlib.modeling import DenseModel
from meshsuggestlib.data import EncodeDataset
from meshsuggestlib.retriever import BaseFaissIPRetriever
from meshsuggestlib.evaluation import Evaluator
from meshsuggestlib.data import EncodeCollator
from meshsuggestlib.suggestion import suggest_mesh_terms,prepare_model
from meshsuggestlib.submission import combine_query, submit_result
from tqdm import tqdm
import torch
from contextlib import nullcontext
import numpy as np
import gc
logger = logging.getLogger(__name__)


def write_ranking(topic, final_result, output):
    for rank, r in enumerate(final_result):
        output.write(f'{topic}\t{r}\t{rank + 1}\n')

def encoding(dataset, model, tokenizer, max_length, hf_args, async_args, encode_is_qry=False):
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
                batch.to(async_args.device)
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
            if os.path.exists(mesh_args.qrel_file):
                Evaluator(mesh_args.qrel_file, ["SetP", "SetR", "SetF(beta=1)", "SetF(beta=3)"], mesh_args.output_file)
                sys.exit()
            else:
                raise Exception("Please put correct qrel file path")
    logger.info("HuggingFace parameters %s", hf_args)
    logger.info("MeSHSuggestLib parameters %s", mesh_args)

    deploy_dataset = mesh_args.dataset
    method = mesh_args.method

    #pre_datasets = ['clef-2017', 'clef-2018', 'clef-2019-dta', 'clef-2019-intervention']
    pre_methods_base = ['ATM', 'MetaMAP', 'UMLS']
    pre_methods_bert = ['Atomic-BERT', 'Semantic-BERT', 'Fragment-BERT']

    mesh_file = mesh_args.mesh_file

    if deploy_dataset in pre_datasets:
        data_parent_folder = 'data/clef-tar-processed/' + deploy_dataset + '/testing/*'
    else:
        data_parent_folder = 'data/' + deploy_dataset + '/*'
    topics_pathes = glob(data_parent_folder)

    input_keywords_dict = {}
    input_clause_dict = {}

    for topic_path in topics_pathes:
        topic = topic_path.split('/')[-1]
        clause_pathes = glob(topic_path+'/*')
        for clause_path in clause_pathes:
            index = topic_path + '_' + clause_path
            clause_num = clause_path.split('/')[-1]
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

    if method in pre_methods_base:
        a = 0 #need to add code here

    if method in pre_methods_bert:
        mesh_dict, tokenizer, model, model_w2v = prepare_model(mesh_args.model_dir, mesh_file, 1, mesh_args.tokenizer_name_or_path, mesh_args.cache_dir, mesh_args.semantic_model_path)
        candidate_dataset = EncodeDataset(mesh_file,
                                          tokenizer,
                                          max_len=mesh_args.p_max_len,
                                          cache_dir=mesh_args.cache_dir)
        p_lookup, p_reps = encoding(candidate_dataset, model, tokenizer, mesh_args.p_max_len, hf_args, mesh_args)
        retriever = BaseFaissIPRetriever(p_reps.float().numpy())

        logger.info("Starts Retrieval")
        try:
            output = open(mesh_args.output_file, 'w')
        except:
            raise Exception("can not create output file at Path %s", mesh_args.output_file)
        final_query_dict = {}
        for topic_index in tqdm(input_keywords_dict):
            input_keywords = input_keywords_dict[topic_index]
            no_mesh_clause = input_keywords[topic_index]
            input_dict = {
                "Keywords": input_keywords,
                "Type": method,
            }
            suggestion_results = suggest_mesh_terms(input_dict, model, tokenizer, retriever, p_lookup, mesh_dict, model_w2v, mesh_args.interpolation_depth, mesh_args.depth)["MeSH_Terms"]
            suggested_mesh_terms = []
            for r in suggestion_results:
                suggested_mesh_terms += r["MeSH_Terms"]
            suggested_mesh_terms = list(set(suggested_mesh_terms))
            new_query = combine_query(no_mesh_clause, suggested_mesh_terms)
            topic_parent = topic_index.split('_')[0]
            if topic_parent not in final_query_dict:
                final_query_dict[topic_parent] = []
            final_query_dict[topic_parent].append(new_query)

        for topic_id in final_query_dict:
            mesh_query = " AND ".join(final_query_dict[topic_id])
            final_result = submit_result(mesh_query, mesh_args.email)
            write_ranking(topic_id, final_result, output)

if __name__ == "__main__":
    main()
