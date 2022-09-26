# meshsuggestlib

Boolean query construction is often critical for medical systematic review literature search. To create an effective Boolean query, systematic review researchers typically spend weeks coming up with effective query terms and combinations. One challenge to creating an effective systematic review Boolean query is to select effective MeSH Terms in the query. In our previous work, we created neural MeSH term suggestion methods and compared them to state-of-the-art MeSH term suggestion methods. 
We found neural MeSH term suggestion methods to be highly effective. 
In this package,  We implement ours and others MeSH term suggestion methods and that is aimed at researchers who want to further investigate, create or deploy such type of methods.

## Setup:
To install our package use:

```
git clone https://github.com/ielab/meshsuggestlib.git
cd meshsuggestlib
pip install .
```

Our package depends on [Tevatron](https://github.com/texttron/tevatron). However, we found we can not use Tevatron version on pip, please use the following command to install editable version:

```
git clone https://github.com/texttron/tevatron
cd tevatron
pip install .
```

Please setup your local enviroment and install all required package using:

```
pip install -r requirements.txt
```


## Preparation
To install our fine-tuned BERT checkpoint, install using:




## Reproduce:

### Input Options:
List of pre-built method option include:
```
- Original
- ATM
- MetaMAP
- UMLS
- Atomic-BERT
- Semantic-BERT
- Fragment-BERT
```

List of pre-defined dataset option include:
```
- CLEF-2017
- CLEF-2018
- CLEF-2019-dta
- CLEF-2019-intervention
```

### Suggestion:

Running the following sample code can output result using CLEF-2017 dataset using Semantic-BERT

```
python -m meshsuggestlib
--output_dir model/
--model_dir model/checkpoint-80000/
--method Semantic-BERT
--dataset CLEF-2017
--output_file result/out.tsv
--email sample@gmail.com
--interpolation_depth 20
--depth 1
```

### Evaluation:

```
python -m meshsuggestlib
--evaluate_run \
--dataset CLEF-2017 \
--output_dir model/
--qrel_file data/clef-tar-processed/CLEF-2017/testing/data.qrels \
--output_file result/out.tsv
```

Preliminary Result to check:

| Dataset       | CLEF-2017                    | CLEF-2018                    | CLEF-2019-dta                    | CLEF-2019-intervention       |
|---------------|------------------------------|------------------------------|----------------------------------|------------------------------|
| Method\Metric | Precision/F_1/Recall         | Precision/F_1/Recall         | Precision/F_1/Recall             | Precision/F_1/Recall         |
| Original      | 0.0303/0.0323/0.7695         | 0.0226/0.0415/**0.8629**     | **0.0246**/**0.0453**/**0.8948** | 0.0166/0.0217/0.7450         |
| ATM           | 0.0225/0.0215/0.7109         | 0.0306/0.0535/0.8224         | 0.0111/0.0207/0.8936             | 0.0155/0.0181/0.7087         |
| MetaMAP       | 0.0323/0.0304/0.7487         | 0.0336/0.0590/0.8085         | 0.0137/0.0254/0.8774             | 0.0187/0.0211/0.6790         |
| UMLS          | 0.0325/0.0300/0.7379         | 0.0325/0.0573/0.7937         | 0.0133/0.0249/0.8598             | 0.0169/0.0186/0.6861         |
| Atomic-BERT   | 0.0252/0.0243/0.7778         | 0.0283/0.0479/0.8452         | 0.0096/0.0180/0.8850             | 0.0062/0.0111/**0.7586**     |
| Semantic-BERT | 0.0255/0.0243/**0.7785**     | 0.0309/0.0526/0.8403         | 0.0108/0.0202/0.8810             | 0.0108/0.0181/0.7507         |
| Fragment-BERT | **0.0343**/**0.0325**/0.7415 | **0.0388**/**0.0690**/0.8034 | 0.0235/0.0364/0.8765             | **0.0224**/**0.0276**/0.7165 |

From the preliminary result it's clear that the best performance almost always in Neural Suggestion Methods.

## Research

### New Dataset:
To use new dataset in our library, build dataset structure as follows:

    .
    ├── ...
    ├── New-DATASET                     # DATASET Folder Name
    │   ├── TOPIC1                      # Topic Name
    │   │     │── 1                     # Clause Number
    │   │     │   │── clause_no_mesh    # MeSH Term removed Clause (one line)
    │   │     │   └── keywords          # keyword in Clause (one keyword per line)
    │   │     │── 2     
    │   │     └── 3
    │   ├── TOPIC2       
    │   └── data.qrels                  # relevance judgement of query documents, format per line {topicid   0   docid   1}
     ...

Then to use new dataset, use ***--dataset DATASET_NAME*** during suggestion.

### New Search Function:

To use new functions, modify *user_defined_method* function in [Suggestion.py](src/meshsuggestlib/suggestion.py) 

Then to use new Search Function, use ***--METHOD NEW*** during suggestion.








