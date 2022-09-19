import ir_measures
import sys
from tqdm import tqdm

removed_topics = ["CD007427", "CD010771", "CD010772", "CD010775", "CD010783", "CD010860", "CD011145",
                  "CD007427", "CD009263", "CD009694",
                  "CD010276",
                  "CD006715", "CD007427", "CD009263", "CD009694", "CD011768"]


def _get_metrics(metrics):
    measures, errors = [], []
    for m in metrics:
        try:
            measure = ir_measures.parse_measure(m)
            if measure not in measures:
                measures.append(measure)
        except ValueError:
            errors.append(f'syntax error: {m}')
        except NameError:
            errors.append(f'unknown metrics: {m}')
    if errors:
        sys.stderr.write('\n'.join(['error parsing metrics'] + errors + ['']))
        sys.exit(-1)
    return measures


class Evaluator:
    def __init__(self, qrel_file, metrics, run_file):
        self.qrels = self._read_qrel(qrel_file)
        self.metrics = _get_metrics(metrics)
        self.run = self.read_run(run_file)


    def compute_metrics(self):
        #print(self.qrels)
        results = ir_measures.calc_aggregate(self.metrics, self.qrels, self.run)
        print(results.items())
        return results.items()
    def read_run(self, run_file):
        run_results = {}
        with open(run_file) as f:
            for l in f:
                try:
                    qid, docid, score = l.strip().split('\t')
                except ValueError:
                    raise ValueError("Wrong run format.")
                if qid in removed_topics:
                    continue
                if qid not in run_results:
                    run_results[qid] = {}
                run_results[qid][docid] = float(score)
        return run_results


    def _read_qrel(self, qrel_file):
        qrels = {}
        with open(qrel_file, 'r') as f:
            for l in f:
                try:
                    qid, _, docid, rel = l.strip().split()
                except ValueError:
                    raise ValueError("Wrong qrel format.")
                if qid in removed_topics:
                    continue
                if (qid not in qrels) and int(rel)!=0:
                    qrels[qid] = {}
                if int(rel)!=0:
                    qrels[qid][docid] = int(rel)
        return qrels
