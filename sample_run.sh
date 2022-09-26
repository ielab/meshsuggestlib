python3 -m meshsuggestlib \
  --output_dir model/ \
  --model_dir model/checkpoint-80000/ \
  --mesh_encoding model/passage.pt \
  --method Fragment-BERT \
  --tokenizer_name_or_path biobert-v1.1/ \
  --dataset CLEF-2019-intervention \
  --semantic_model_path model/PubMed-w2v.bin \
  --cache_dir cache/ \
  --output_file result/fragment_2019_intervention.tsv \
  --qrel_file data/clef-tar-processed/CLEF-2019-intervention/testing/data.qrels \
  --email wshuai190@gmail.com \
  --interpolation_depth 20 \
  --depth 1


python3 -m meshsuggestlib \
--evaluate_run \
--dataset CLEF-2019-dta \
--output_dir model/ \
--qrel_file data/clef-tar-processed/CLEF-2019-intervention/testing/data.qrels \
--output_file result/UMLS_2019_intervention.tsv


