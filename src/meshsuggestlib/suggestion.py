from transformers import AutoConfig, AutoTokenizer
from meshsuggestlib.modeling import DenseModel
import os
import torch
import json
from gensim.models import KeyedVectors
from gensim.utils import tokenize
import numpy
from contextlib import nullcontext
import scipy
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
from abc import ABC, abstractmethod
import subprocess
import requests

def load_mesh_dict(path):
    mesh_dict = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            uid = item['text_id']
            original_term = item['text']
            mesh_dict[uid] = original_term
    return mesh_dict
def get_mesh_terms(uids, mesh_dict):
    mesh_terms = {index: mesh_dict[uid] for index, uid in enumerate(uids) if uid in mesh_dict}
    return mesh_terms

class ATM_Suggest:
    def __init__(self, mesh_args):
        self.url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.key = mesh_args.atm_key

    def suggest(self, input_dict):
        terms = input_dict["Keywords"]
        result = []
        for term in terms:
            mesh_for_single_term = {
                "Keywords": [term],
                "type": "ATM",
                "MeSH_Terms": {}
            }
            url = f'{self.url}?db=pubmed&api_key={self.key}&retmode=json&term={term}'
            response = requests.get(url)
            content = json.loads(response.content)
            translation_stack = content["esearchresult"]["translationset"]
            if len(translation_stack)==0:
                continue
            for item in translation_stack:
                translated_terms = item['to'].split("OR")
                for t in translated_terms:
                    t = t.strip()
                    if t.endswith("[MeSH Terms]"):
                        for char in ['*', '"', '[MeSH Terms]', '"', '"']:
                            mesh = t.replace(char, "")
                        mesh_for_single_term['MeSH_Terms'][len(mesh_for_single_term['MeSH_Terms'])] = mesh
                result.append(mesh_for_single_term)
        return result

class UMLS_MeSH_Suggestion:
    def __init__(self):
        self.base_url = "http://127.0.0.1:9200/umls/_search?pretty=true&q="

    def suggest(self, input_dict):
        terms = input_dict["Keywords"]
        result = []

        for term in terms:
            umls_terms = set()
            res = requests.get(self.base_url + term)
            dict_set = json.loads(res.text)
            words = dict_set["hits"]["hits"]
            for word in words:
                #score = word["_score"]
                sources = word["_source"]["thesaurus"]
                for source in sources:
                    if "MRCONSO_STR" in source:
                        type = source['MRCONSO_SAB']
                        if type=="MSH":
                            mesh_term = source["MRCONSO_STR"]
                            umls_terms.add(mesh_term)
            m_dict = {i:t for i, t in enumerate(umls_terms)}
            mesh_for_single_term = {
                "Keywords": [term],
                "type": "UMLS",
                "MeSH_Terms": m_dict
            }

            result.append(mesh_for_single_term)
        return result

class MetaMap_MeSH_Suggestion:
    def __init__(self):
        self.base_url = "http://127.0.0.1:9200/umls/_search?pretty=true&q="

    def suggest(self, input_dict):
        terms = input_dict["Keywords"]
        result = []
        for term in terms:
            umls_terms = set()
            metamap_get_terms = subprocess.Popen('echo "' + term + '" | public_mm/bin/metamap -I', shell=True, stdout=subprocess.PIPE).stdout
            metamap_terms = metamap_get_terms.read().decode().split('\n')

            for metamap_term in metamap_terms:
                chunks = metamap_term.split()
                for chunk in chunks:
                    if (":" in chunk) and ("C" in chunk):
                        term_id = chunk.split(":")[0]
                        print(term_id)
                        res = requests.get(self.base_url + "cui:" + term_id)
                        print(res)
                        dict_set = json.loads(res.text)
                        words = dict_set["hits"]["hits"]
                        for word in words:
                            # score = word["_score"]
                            sources = word["_source"]["thesaurus"]
                            for source in sources:
                                if "MRCONSO_STR" in source:
                                    type = source['MRCONSO_SAB']
                                    if type == "MSH":
                                        mesh_term = source["MRCONSO_STR"]
                                        umls_terms.add(mesh_term)
            m_dict = {i: t for i, t in enumerate(umls_terms)}
            mesh_for_single_term = {
                "Keywords": [term],
                "type": "ATM",
                "MeSH_Terms": m_dict
            }
            result.append(mesh_for_single_term)
        return result

class NeuralSuggest:
    def __init__(self, mesh_args, hf_args):
        self.mesh_args = mesh_args
        self.hf_args = hf_args
        self.mesh_dict, self.tokenizer, self.model, self.model_w2v = self.prepare_model(mesh_args.model_dir, mesh_args.mesh_file, mesh_args.tokenizer_name_or_path, mesh_args.cache_dir, mesh_args.semantic_model_path)



    def prepare_model(self, model_dir, mesh_dir, tokenizer_path, cache_path, semantic_model_path):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        # load mesh_dict
        logger.info("loading mesh terms from Path %s", mesh_dir)
        mesh_dict = load_mesh_dict(mesh_dir)

        # load_model_for_query_encoding

        logger.info("loading models from Path %s", model_dir)

        model = DenseModel(
            ckpt_path=model_dir,
            mesh_args=self.mesh_args,
        ).eval()

        model = model.to(self.mesh_args.device)

        logger.info("loading tokenizer from %s", tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_path)

        logger.info("loading semantic model from %s", semantic_model_path)
        if semantic_model_path!= None:
            model_w2v = KeyedVectors.load_word2vec_format(semantic_model_path, binary=True)
        else:
            model_w2v = None

        return mesh_dict, tokenizer, model, model_w2v

    def user_defined_method(self, keywords, retriever, look_up):
        suggestion_uids = []
        raise NotImplementedError()
        return suggestion_uids

    def suggest_mesh_terms(self, input_dict, retriever, look_up):
        type = input_dict["Type"]
        keywords = input_dict["Keywords"]
        if len(keywords) > 0:
            suggestion_comb = []
            if type == "Atomic-BERT":
                for keyword in keywords:
                    suggestion_uids = self.keyword_suggestion_method(keyword, retriever, look_up)
                    suggestion_comb.append(([keyword], suggestion_uids))
            elif type == "Semantic-BERT":
                keyword_groups = self.seperate_keywords_group(keywords)
                for keywords in keyword_groups:
                    suggestion_uids = self.semantic_suggestion_method(keywords, retriever, look_up)
                    suggestion_comb.append((keywords, suggestion_uids))
            elif type == "Fragment-BERT":
                suggestion_uids = self.fragment_suggestion_method(keywords, retriever, look_up)
                suggestion_comb.append((keywords, suggestion_uids))
            else:
                suggestion_comb = self.user_defined_method(keywords, retriever, look_up)
            return_list = []
            for keyword_groups, suggestion_uids in suggestion_comb:
                mesh_terms = get_mesh_terms(suggestion_uids, self.mesh_dict)
                new_dict = {
                    "Keywords": keyword_groups,
                    "type": type,
                    "MeSH_Terms": mesh_terms
                }
                return_list.append(new_dict)
            return return_list
        else:
            raise Exception("Minimum one keyword to suggest")

    def search_queries(self, retriever, q_rep, lookup):
        all_scores, all_indices = retriever.search(q_rep, self.mesh_args.depth)
        psg_indices = [[str(lookup[x]) for x in q_dd] for q_dd in all_indices]
        return psg_indices

    def search_queries_multiple(self, retriever, q_reps, lookup):
        returned_indices = []
        overall_psg_indices = {}
        for q_rep in q_reps:
            all_scores, all_indices = retriever.search(q_rep, self.mesh_args.interpolation_depth)
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
            if count>self.mesh_args.depth:
                break
            prev = sorted_item[1]
            returned_indices.append(sorted_item[0])
        return returned_indices


    def seperate_keywords_group(self, keywords):
        keywords = [k.lower() for k in keywords]
        key_ids = []
        query_vectors = []
        keyword_groups = []
        for key_index, k in enumerate(keywords):
            model_incidence = [self.model_w2v[token] for token in tokenize(k) if token in self.model_w2v]
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

    def keyword_suggestion_method(self, keyword, retriever, look_up):
        query = keyword.lower()
        query_tokenised = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=self.mesh_args.q_max_len,
            truncation=True,
            padding=False,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        with torch.cuda.amp.autocast() if self.hf_args.fp16 else nullcontext():
            with torch.no_grad():
                query_tokenised.to(self.mesh_args.device)
                q_reps = self.model.encode_query(query_tokenised)
                encoded = q_reps.cpu().detach().numpy()
                uids = self.search_queries(retriever, encoded, look_up)
        return uids[0]


    def semantic_suggestion_method(self, keywords,retriever, look_up):
        encoded = []
        for keyword in keywords:
            query = keyword.lower()
            query_tokenised = self.tokenizer.encode_plus(
                query,
                add_special_tokens=True,
                max_length=self.mesh_args.q_max_len,
                truncation=True,
                padding=False,
                return_token_type_ids=False,
                return_attention_mask=True,
                return_tensors='pt'
            )
            with torch.cuda.amp.autocast() if self.hf_args.fp16 else nullcontext():
                with torch.no_grad():
                    query_tokenised.to(self.mesh_args.device)
                    q_reps = self.model.encode_query(query_tokenised)
                    encoded.append(q_reps.cpu().detach().numpy())
        if len(encoded) > 1:
            uids = [self.search_queries_multiple(retriever, encoded, look_up)]
        else:
            uids = self.search_queries(retriever, encoded[0], look_up)
        return uids[0]


    def fragment_suggestion_method(self, keywords,retriever, look_up):
        encoded = []
        for keyword in keywords:
            query = keyword.lower()
            query_tokenised = self.tokenizer.encode_plus(
                query,
                add_special_tokens=True,
                max_length=self.mesh_args.q_max_len,
                truncation=True,
                padding=False,
                return_token_type_ids=False,
                return_attention_mask=True,
                return_tensors='pt'
            )
            with torch.cuda.amp.autocast() if self.hf_args.fp16 else nullcontext():
                with torch.no_grad():
                    query_tokenised.to(self.mesh_args.device)
                    q_reps = self.model.encode_query(query_tokenised)
                    encoded.append(q_reps.cpu().detach().numpy())
        uids = [self.search_queries_multiple(retriever, encoded, look_up)]
        return uids[0]
