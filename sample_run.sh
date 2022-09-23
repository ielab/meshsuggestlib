#python3 -m meshsuggestlib \
#  --output_dir model/ \
#  --model_dir model/checkpoint-80000/ \
#  --mesh_encoding model/passage.pt \
#  --method  Original \
#  --tokenizer_name_or_path dmis-lab/biobert-v1.1 \
#  --dataset CLEF-2018 \
#  --semantic_model_path model/PubMed-w2v.bin \
#  --cache_dir cache/ \
#  --fp16 \
#  --output_file result/original_2018.tsv \
#  --qrel_file /scratch/itee/uqswan37/meshsuggestlib/data/clef-tar-processed/CLEF-2018/testing/data.qrels \
#  --email wshuai190@gmail.com \
#  --interpolation_depth 20 \
#  --depth 1


python3 -m meshsuggestlib \
--evaluate_run \
--dataset CLEF-2018 \
--method MetaMAP \
--output_dir model/ \
--qrel_file data/clef-tar-processed/CLEF-2018/testing/data.qrels \
--output_file result/MetaMap_2018.tsv


