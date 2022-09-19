import torch
from typing import Dict
from torch import Tensor
from meshsuggestlib.arguments import MeSHSuggestLibArguments
import logging
logger = logging.getLogger(__name__)
try:
    from tevatron.modeling import DenseModel as DenseModelForInference
    HAS_TEVATRON = True
except ModuleNotFoundError as err:
    logger.warning("Cannot find tevatron package.")
    HAS_TEVATRON = False



class Encoder(torch.nn.Module):
    def __init__(self, ckpt_path: str, mesh_args: MeSHSuggestLibArguments):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.mesh_args = mesh_args

    @torch.no_grad()
    def encode_passage(self, psg: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("Please implement DenseModel class in '/meshsuggestlib/src/meshsuggestlib/modeling.py'")

    @torch.no_grad()
    def encode_query(self, qry: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("Please implement DenseModel class in '/meshsuggestlib/src/meshsuggestlib/modeling.py'")


class DenseModel(Encoder):
    """
    For validating customized dense retriever checkpoints, please modify this class to meet your encoding requirements.
    """
    def __init__(self, ckpt_path, mesh_args):
        super(DenseModel, self).__init__(ckpt_path, mesh_args)
        if HAS_TEVATRON:
            self.model = DenseModelForInference.load(model_name_or_path=self.ckpt_path,
                                                      cache_dir=self.mesh_args.cache_dir)

    def encode_passage(self, psg):
        if HAS_TEVATRON:
            return self.model.encode_passage(psg)
        else:
            return super(DenseModel, self).encode_passage(psg)

    def encode_query(self, qry):
        if HAS_TEVATRON:
            return self.model.encode_passage(qry)
        else:
            return super(DenseModel, self).encode_passage(qry)