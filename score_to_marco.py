# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import defaultdict

removed_topics = ["CD007427", "CD010771", "CD010772", "CD010775", "CD010783", "CD010860", "CD011145",
                  "CD007427", "CD009263", "CD009694",
                  "CD010276",
                  "CD006715", "CD007427", "CD009263", "CD009694", "CD011768"]

parser = argparse.ArgumentParser()
parser.add_argument('--score_file', required=True)
parser.add_argument('--run_id', default='marco')
args = parser.parse_args()

with open(args.score_file) as f:
    lines = f.readlines()

all_scores = defaultdict(dict)

for line in lines:
    if len(line.strip()) == 0:
        continue
    qid, did, score = line.strip().split()
    score = float(score)
    all_scores[qid][did] = score

qq = list(all_scores.keys())

with open(args.score_file + '.trec', 'w') as f:
    for qid in qq:
        score_list = sorted(list(all_scores[qid].items()), key=lambda x: x[1], reverse=True)
        for rank, (did, score) in enumerate(score_list):
            if qid in removed_topics:
                continue
            f.write(f'{qid} Q0 {did} {rank+1} {score} Interpolated\n')

