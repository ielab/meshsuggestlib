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




## To reproduce result on Pre-built MeSH Term Suggestion methotds. run:

```
python -m meshsuggestlib
--output_dir model/
--model_dir model/checkpoint-80000/
--method Semantic-BERT
--dataset CLEF-2017
--output_file out.tsv
--email sample@gmail.com
--interpolation_depth 20
--depth 1
```
