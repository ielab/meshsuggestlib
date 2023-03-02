python3 -m meshsuggestlib \
  --output_dir model/ \
  --method ATM \
  --dataset clef-tar-processed/new_query \
  --cache_dir cache/ \
  --output_file result/new_query_atm.tsv \
  --qrel_file data/clef-tar-processed/CLEF-2017/train/data.qrels \
  --email wshuai190@gmail.com \
  --atm_key f4bc8cc0e17e8c4328c18ff0a89b5328d108 \
  --depth 1 \
  --device cpu

python3 -m meshsuggestlib \
  --output_dir model/ \
  --method MetaMAP \
  --dataset clef-tar-processed/new_query \
  --cache_dir cache/ \
  --output_file result/new_query_metamap.tsv \
  --qrel_file data/clef-tar-processed/CLEF-2017/train/data.qrels \
  --depth 1 \
  --device cpu


python3 -m meshsuggestlib \
  --output_dir model/ \
  --method UMLS \
  --dataset clef-tar-processed/new_query \
  --cache_dir cache/ \
  --output_file result/new_query_umls.tsv \
  --qrel_file data/clef-tar-processed/CLEF-2017/train/data.qrels \
  --depth 1 \
  --device cpu

python3 -m meshsuggestlib \
  --output_dir model/ \
  --method Atomic-BERT \
  --dataset clef-tar-processed/new_query \
  --model_dir model/checkpoint-80000/ \
  --tokenizer_name_or_path dmis-lab/biobert-v1.1 \
  --cache_dir cache/ \
  --output_file result/new_query_atomic.tsv \
  --qrel_file data/clef-tar-processed/CLEF-2017/train/data.qrels \
  --email wshuai190@gmail.com \
  --email wshuai190@gmail.com \
  --depth 1 \
  --device cpu

python3 -m meshsuggestlib \
  --output_dir model/ \
  --method Semantic-BERT \
  --dataset clef-tar-processed/new_query \
  --model_dir model/checkpoint-80000/ \
  --tokenizer_name_or_path dmis-lab/biobert-v1.1 \
  --semantic_model_path model/PubMed-w2v.bin \
  --cache_dir cache/ \
  --output_file result/new_query_semantic.tsv \
  --qrel_file data/clef-tar-processed/CLEF-2017/train/data.qrels \
  --email wshuai190@gmail.com \
  --interpolation_depth 20 \
  --depth 1 \
  --device cpu

python3 -m meshsuggestlib \
  --output_dir model/ \
  --method Fragment-BERT \
  --dataset clef-tar-processed/new_query \
  --model_dir model/checkpoint-80000/ \
  --tokenizer_name_or_path dmis-lab/biobert-v1.1 \
  --cache_dir cache/ \
  --output_file result/new_query_fragment.tsv \
  --qrel_file data/clef-tar-processed/CLEF-2017/train/data.qrels \
  --email wshuai190@gmail.com \
  --depth 1 \
  --interpolation_depth 20 \
  --device cpu









#python3 -m meshsuggestlib \
#--evaluate_run \
#--output_dir model/ \
#--qrel_file data/clef-tar-processed/CLEF-2018/testing/data.qrels \
#--output_file result/semantic_bert_2018_2.tsv


