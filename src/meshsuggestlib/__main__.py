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
from tqdm import tqdm
import torch
from contextlib import nullcontext
import numpy as np
import gc
logger = logging.getLogger(__name__)


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for rank, (s, idx) in enumerate(score_list):
                f.write(f'{qid}\t{idx}\t{rank + 1}\n')


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

    tokenizer = AutoTokenizer.from_pretrained(
        mesh_args.tokenizer_name_or_path,
        cache_dir=mesh_args.cache_dir,
        use_fast=True,
    )

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
            with open(keyword_file, 'r') as f:
                for l in f:



    if method in pre_methods_base:
        a = 0 #need to add code here

    if method in pre_methods_bert:
        model = DenseModel(
            model_path=model_dir,
            mesh_args=mesh_args,
        ).eval()
        model = model.to(mesh_args.device)
        p_lookup, p_reps = encoding(candidate_dataset, model, tokenizer, async_args.p_max_len, hf_args, async_args)


    for query_file in async_args.query_files:
        query_dataset = EncodeDataset(query_file,
                                      tokenizer,
                                      max_len=async_args.q_max_len,
                                      cache_dir=async_args.cache_dir)
        query_datasets.append(query_dataset)

    files = os.listdir(async_args.candidate_dir)
    candidate_files = [
        os.path.join(async_args.candidate_dir, f) for f in files if f.endswith('json')
    ]
    candidate_dataset = EncodeDataset(candidate_files,
                                      tokenizer,
                                      max_len=async_args.p_max_len,
                                      cache_dir=async_args.cache_dir)

    evaluator = Evaluator(async_args.qrel_file, async_args.metrics)

    finished_ckpts = set()
    last_change = os.stat(async_args.ckpts_dir).st_mtime

    hit_max = False
    logger.info(f"Listening to {async_args.ckpts_dir}")
    # Validation loop
    while not hit_max:
        current_change = os.stat(async_args.ckpts_dir).st_mtime
        if current_change != last_change or len(os.listdir(async_args.ckpts_dir)) != 0:
            time.sleep(5)
            last_change = current_change
            ckpt_time = [(ckpt, os.stat(os.path.join(async_args.ckpts_dir, ckpt)).st_mtime)
                         for ckpt in os.listdir(async_args.ckpts_dir)]
            ckpt_time = sorted(ckpt_time, key=lambda x: x[1])

            # validation loop
            for ckpt, _ in ckpt_time:
                ckpt_path = os.path.join(async_args.ckpts_dir, ckpt)
                if ckpt_path in finished_ckpts:
                    continue
                logger.info(f"Evaluating {ckpt_path}")

                model = DenseModel(
                    ckpt_path=ckpt_path,
                    async_args=async_args,
                ).eval()
                model = model.to(async_args.device)

                p_lookup, p_reps = encoding(candidate_dataset, model, tokenizer, async_args.p_max_len, hf_args, async_args)

                q_lookup_list = []
                q_reps_list = []
                for query_dataset in query_datasets:
                    q_lookup, q_reps = encoding(query_dataset, model, tokenizer, async_args.q_max_len,
                                                hf_args, async_args, encode_is_qry=True)
                    q_lookup_list.append(q_lookup)
                    q_reps_list.append(q_reps)

                retriever = BaseFaissIPRetriever(p_reps.float().numpy())

                all_scores_list = []
                psg_indices_list = []
                for q_reps in q_reps_list:
                    all_scores, psg_indices = search_queries(retriever, q_reps.float().numpy(), p_lookup, async_args)
                    all_scores_list.append(all_scores)
                    psg_indices_list.append(psg_indices)

                for i, (all_scores, psg_indices, q_lookup) in enumerate(zip(all_scores_list,
                                                                        psg_indices_list,
                                                                        q_lookup_list)):
                    if async_args.write_run:
                        if not os.path.exists(hf_args.output_dir):
                            os.makedirs(hf_args.output_dir)
                        write_ranking(psg_indices, all_scores, q_lookup, os.path.join(hf_args.output_dir, f'set_{i}_{ckpt}.tsv'))



if __name__ == "__main__":
    main()
