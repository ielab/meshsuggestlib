import numpy as np
import math
from cwl.ruler.measures.cwl_metrics import CWLMetric

"""
Discounted Cumulative Gain by Jarvelin and Kekalainen (2002)
The discount is SCALED so that forms a proper probability distribution.

k is the rank cut off i.e number of items to be examined
base is the base of the log for the discounting, which is set to 2 by default as per the original paper.

Note that the scaled DCG under CWL is equivalent to NDCG when the ideal ranking is composed of a vector of max gains.
That is sDCG implicitly assumes that there is an infinite number of max gain items, and so tends to provide a more
conservative estimate of the gain than the original implementation of NDCG.

Essentially the DCG metrics in CWL only approximates NDCG, and under certain conditions is equivalent.
So it is better to refer to scaled DCG as opposed to NDCG.
"""


class SDCGCWLMetric(CWLMetric):
    def __init__(self, k):
        super().__init__()
        self.metric_name = "sDCG-k@{0}".format(k)
        self.k = k
        self.base = 2.0
        self.bibtex = """
        @article{Jarvelin:2002:CGE:582415.582418,
        author = {J\"{a}rvelin, Kalervo and Kek\"{a}l\"{a}inen, Jaana},
        title = {Cumulated Gain-based Evaluation of IR Techniques},
        journal = {ACM Trans. Inf. Syst.},
        volume = {20},
        number = {4},
        year = {2002},
        pages = {422--446},
        numpages = {25},
        url = {http://doi.acm.org/10.1145/582415.582418},
        }
        """

    def name(self):
        return "sDCG-k@{0}".format(self.k)

    def c_vector(self, ranking, worse_case=True):

        cvec = []
        for i in range(1, ranking.n+1):
            if i < self.k:
                cvec.append(math.log(i+1, self.base)/math.log(i+2, self.base))
            else:
                cvec.append(0.0)

        cvec = np.array(cvec)

        return cvec


class NDCGCWLMetric(SDCGCWLMetric):

    def __init__(self, k):
        super().__init__(k)
