python3 -m meshsuggestlib \
  --output_dir model/ \
  --model_dir model/checkpoint-80000/ \
  --mesh_encoding model/passage.pt \
  --method Fragment-BERT \
  --tokenizer_name_or_path dmis-lab/biobert-v1.1 \
  --dataset CLEF-2017 \
  --semantic_model_path model/PubMed-w2v.bin \
  --cache_dir cache/ \
  --fp16 \
  --output_file result/fragment_bert_2017.tsv \
  --evaluate_run \
  --qrel_file /scratch/itee/uqswan37/meshsuggestlib/data/clef-tar-processed/CLEF-2017/testing/data.qrels \
  --email wshuai190@gmail.com \
  --interpolation_depth 20 \
  --depth 1

