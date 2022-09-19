from dataclasses import dataclass, field
from typing import List


@dataclass
class MeSHSuggestLibArguments:
    model_dir: str = field(
        default=None, metadata={"help": "Path to the folder with model."}
    )
    method: str = field(
        default=None, metadata={"help": "retrieval method, can be Atomic-BERT, Semantic-BERT, Fragment-BERT, ATM, MetaMAP. UMLS or New_Method_Name"}
    )

    tokenizer_name_or_path: str = field(
        default=None, metadata={"help": "Tokenizer"}
    )

    mesh_file: str = field(
        default="data/mesh.jsonl", metadata={
            "help": "Used dataset, currently supported options are 'CLEF-2017', 'CLEF-2018', 'CLEF-2019-dta' and ', 'CLEF-2019-intervention' or dataset FOLDER-NAME"}
    )

    dataset: str = field(
        default=None, metadata={"help": "Used dataset, currently supported options are 'CLEF-2017', 'CLEF-2018', 'CLEF-2019-dta' and ', 'CLEF-2019-intervention' or dataset FOLDER-NAME"}
    )

    semantic_model_path: str = field(
        default=None, metadata={"help": "Cache folder."}
    )

    cache_dir: str = field(
        default=None, metadata={"help": "Cache folder."}
    )

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    m_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for MeSH Terms. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    retrieve_batch_size: int = field(default=64, metadata={"help": "Faiss retrieval batch size."})

    depth: int = field(default=100, metadata={"help": "Final Suggestion depth."})

    interpolation_depth: int = field(default=100, metadata={"help": "Interpolation depth for"})

    evaluate_run: bool = field(
        default=True, metadata={"help": "Evaluate the methods or not"}
    )

    qrel_file: str = field(
        default=None, metadata={"help": "Path to the output file of query"}
    )

    output_file: str = field(
        default=None, metadata={"help": "Path to the output file of query"}
    )

    email: str = field(
        default='email', metadata={"help": "The GPU device uses for encoding"}
    )
    device: str = field(
        default='cuda:0', metadata={"help": "The GPU device uses for encoding"}
    )