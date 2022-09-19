from meshsuggestlib.retriever import BaseFaissIPRetriever
from transformers import AutoConfig, AutoTokenizer
from meshsuggestlib.modeling import DenseModel
import os
import torch
from itertools import chain
import json
from gensim.models import KeyedVectors
from gensim.utils import tokenize
import numpy
from contextlib import nullcontext
import scipy
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm


def suggest_mesh_terms(input_dict, model, tokenizer, retriever, look_up, mesh_dict, model_w2v, interpolation_depth, depth, hf_args, device):
    type = input_dict["Type"]
    keywords = input_dict["Keywords"]
    if len(keywords) > 0:
        return_list = []
        if len(keywords) == 1:
            type = "Atomic-BERT"
        if type == "Atomic-BERT":
            for keyword in keywords:
                suggestion_uids = keyword_suggestion_method(keyword, model, tokenizer, retriever, look_up, depth, hf_args, device)
                mesh_terms = get_mesh_terms(suggestion_uids, mesh_dict)
                new_dict = {
                    "Keywords": [keyword],
                    "type": type,
                    "MeSH_Terms": mesh_terms
                }
                return_list.append(new_dict)
            return return_list
        elif type == "Semantic-BERT":
            keyword_groups = seperate_keywords_group(keywords, model_w2v)
            for keywords in keyword_groups:
                suggestion_uids = semantic_suggestion_method(keywords, model, tokenizer, retriever, look_up, interpolation_depth, depth, hf_args, device)
                mesh_terms = get_mesh_terms(suggestion_uids, mesh_dict)
                new_dict = {
                    "Keywords": keywords,
                    "type": type,
                    "MeSH_Terms": mesh_terms
                }
                return_list.append(new_dict)
            return return_list
        elif type == "Fragment-BERT":
            suggestion_uids = fragment_suggestion_method(keywords, model, tokenizer, retriever, look_up, interpolation_depth, depth, hf_args, device)
            mesh_terms = get_mesh_terms(suggestion_uids, mesh_dict)
            new_dict = {
                "Keywords": keywords,
                "type": type,
                "MeSH_Terms": mesh_terms
            }
            return_list.append(new_dict)
            return return_list

        else:
            raise Exception("Type not valid")
    else:
        raise Exception("Minimum one keyword to suggest")


def load_mesh_dict(path):
    mesh_dict = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            uid = item['text_id']
            original_term = item['text']
            mesh_dict[uid] = original_term
    return mesh_dict


def prepare_model(model_dir, mesh_dir, tokenizer_path, cache_path, semantic_model_path, mesh_args):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # load mesh_dict
    logger.info("loading mesh terms from Path %s", mesh_dir)
    mesh_dict = load_mesh_dict(mesh_dir)

    # load_model_for_query_encoding

    logger.info("loading models from Path %s", model_dir)

    model = DenseModel(
        ckpt_path=model_dir,
        mesh_args=mesh_args,
    )

    model = model.to(mesh_args.device)

    logger.info("loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_path)

    # load_mesh_terms_encoded_and look_ups
    #look_up = []
    # p_reps, p_lookup = pickle_load("data/Encoding/passage.pt")
    # retriever = BaseFaissIPRetriever(p_reps)
    # shards = chain([(p_reps, p_lookup)])
    # for p_reps, p_lookup in shards:
    #     retriever.add(p_reps)
    #     look_up += p_lookup

    logger.info("loading semantic model from %s", semantic_model_path)
    if semantic_model_path!= None:
        model_w2v = KeyedVectors.load_word2vec_format(semantic_model_path, binary=True)
    else:
        model_w2v = None

    return mesh_dict, tokenizer, model, model_w2v


def get_mesh_terms(uids, mesh_dict):
    mesh_terms = {index: mesh_dict[uid] for index, uid in enumerate(uids) if uid in mesh_dict}
    return mesh_terms


def search_queries(retriever, q_rep, lookup, depth):
    all_scores, all_indices = retriever.search(q_rep, depth)
    psg_indices = [[str(lookup[x]) for x in q_dd] for q_dd in all_indices]
    return psg_indices


def search_queries_multiple(retriever, q_reps, lookup, interpolation_depth, depth):
    returned_indices = []
    overall_psg_indices = {}
    for q_rep in q_reps:
        all_scores, all_indices = retriever.search(q_rep, interpolation_depth)
        all_scores = all_scores[0]
        psg_indices = [str(lookup[x]) for x in all_indices[0]]
        min_score = min(all_scores)
        diff_score = max(all_scores) - min(all_scores)
        if diff_score == 0:
            for i, p in psg_indices:
                if psg_indices[i] not in overall_psg_indices:
                    overall_psg_indices[psg_indices[i]] = 0

        for i, s in enumerate(all_scores):
            if psg_indices[i] not in overall_psg_indices:
                overall_psg_indices[psg_indices[i]] = 0
            normalised_score = (all_scores[i] - min_score) / diff_score
            overall_psg_indices[psg_indices[i]] += normalised_score
    sorted_dict = sorted(overall_psg_indices.items(), key=lambda x: x[1], reverse=True)
    count = 0
    prev = 0
    for sorted_item in sorted_dict:
        if sorted_item[1] != prev:
            count+=1
        if count>depth:
            break
        prev = sorted_item[1]
        returned_indices.append(sorted_item[0])
    return returned_indices


def seperate_keywords_group(keywords, model_w2v):
    keywords = [k.lower() for k in keywords]
    key_ids = []
    query_vectors = []
    keyword_groups = []
    for key_index, k in enumerate(keywords):
        model_incidence = [model_w2v[token] for token in tokenize(k) if token in model_w2v]
        if len(model_incidence) >= 1:
            add_vector = numpy.average(model_incidence, axis=0)
            a = numpy.sum(add_vector)
            if not numpy.isnan(a):
                query_vectors.append(add_vector)
                key_ids.append(key_index)
    if len(key_ids) > 1:
        pairs = {}
        for i in range(0, len(key_ids)):
            score = [s[0] for s in scipy.spatial.distance.cdist(query_vectors, [query_vectors[i]], 'cosine')]
            for s_index, s in enumerate(score):
                if (s <= 0.2) and (s_index > i):
                    if key_ids[i] in pairs:
                        pairs[key_ids[i]].append(key_ids[s_index])
                    else:
                        exist = False
                        for current_p in pairs:
                            current_values = pairs[current_p]
                            if (key_ids[i] in current_values) and (key_ids[s_index] in current_values):
                                exist = True
                                break
                            elif (key_ids[i] in current_values):
                                pairs[current_p].append(key_ids[s_index])
                                exist = True
                                break
                            elif (key_ids[s_index] in current_values):
                                pairs[current_p].append(key_ids[i])
                                exist = True
                                break
                        if exist == False:
                            pairs[key_ids[i]] = [key_ids[s_index]]
        already_appeared = set()
        for p in pairs:
            local_pairs = [keywords[p]]
            already_appeared.add(p)
            for a_p in pairs[p]:
                already_appeared.add(a_p)
                local_pairs.append(keywords[a_p])
            keyword_groups.append(local_pairs)
        for id in key_ids:
            if id not in already_appeared:
                keyword_groups.append([keywords[id]])
    else:
        keyword_groups = [[k] for k in keywords]

    return keyword_groups


def keyword_suggestion_method(keyword, model, tokenizer, retriever, look_up, depth, hf_args, device):
    query = keyword.lower()
    query_tokenised = tokenizer.encode_plus(
        query,
        add_special_tokens=False,
        max_length=32,
        truncation=True,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.cuda.amp.autocast() if hf_args.fp16 else nullcontext():
        with torch.no_grad():
            query_tokenised.to(device)
            q_reps = model.encode_query(query_tokenised)
            encoded = q_reps.cpu().detach().numpy()
            uids = search_queries(retriever, encoded, look_up, depth)
    return uids[0]


def semantic_suggestion_method(keywords, model, tokenizer, retriever, look_up, interpolation_depth, depth, hf_args, device):
    encoded = []
    for keyword in keywords:
        query = keyword.lower()
        query_tokenised = tokenizer.encode_plus(
            query,
            add_special_tokens=False,
            max_length=32,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        with torch.cuda.amp.autocast() if hf_args.fp16 else nullcontext():
            with torch.no_grad():
                query_tokenised.to(device)
                q_reps = model.encode_query(query_tokenised)
                encoded.append(q_reps.cpu().detach().numpy())
    if len(encoded) > 1:
        uids = [search_queries_multiple(retriever, encoded, look_up, interpolation_depth, depth)]
    else:
        uids = search_queries(retriever, encoded[0], look_up, depth)

    return uids[0]


def fragment_suggestion_method(keywords, model, tokenizer, retriever, look_up, interpolation_depth, depth, hf_args, device):
    encoded = []
    for keyword in keywords:
        query = keyword.lower()
        query_tokenised = tokenizer.encode_plus(
            query,
            add_special_tokens=False,
            max_length=32,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        with torch.cuda.amp.autocast() if hf_args.fp16 else nullcontext():
            with torch.no_grad():
                query_tokenised.to(device)
                q_reps = model.encode_query(query_tokenised)
                encoded.append(q_reps.cpu().detach().numpy())

    uids = [search_queries_multiple(retriever, encoded, look_up, interpolation_depth, depth)]
    return uids[0]


if __name__ == '__main__':
    mesh_dict, model, tokenizer, retriever, look_up, model_w2v = prepare_model()
    input_dict = {
        "Keywords": ["disease", "Heart attack", "medical condition", "blood test", "blood sample test"],
        "Type": "Semantic"
    }
    print(suggest_mesh_terms(input_dict, model, tokenizer, retriever, look_up, mesh_dict, model_w2v))
