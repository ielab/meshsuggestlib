# elastic-umls

This repository contains nasty, hardcoded, bodged scripts to index the MRCONSO, MEDEF, MDRSTY, and MRREL tables into Elasticsearch.

## Description of scripts

First, download the [UMLS metathesaurus files](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) and unzip them. 

Then, run `create_sqlite.sh`. The first argument to this script must be the extracted folder. This script will import the tables into a sqlite database.

Finally, run `index_elasticsearch.py`. This will query the database and perform bulk inserts into an Elasticsearch index.
**Note**: this script defaults to Elasticsearch running on `http://localhost:9200`. You will need to edit this if you are running Elasticsearch somewhere else.

## Querying Elasticsearch

The index mapping for Elasticsearch is as follows:

```json
{"umls":{"mappings":{"properties":{"cui":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"relations":{"properties":{"AUI1":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"AUI2":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"CUI1":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"CUI2":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"CVF":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"DIR":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"REL":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"RELA":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"RG":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"RUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"SAB":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"SL":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"SRUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"STYPE1":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"STYPE2":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"SUPPRESS":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}}}},"semtypes":{"properties":{"ATUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"CUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"CVF":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"STN":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"STY":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"TUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}}}},"thesaurus":{"properties":{"MRCONSO_AUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_CODE":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_CUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_CVF":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_ISPREF":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_LAT":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_LUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_SAB":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_SAUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_SCUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_SDUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_SRL":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_STR":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_STT":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_SUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_SUPPRESS":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_TS":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRCONSO_TTY":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRDEF_ATUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRDEF_AUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRDEF_CUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRDEF_CVF":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRDEF_DEF":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRDEF_SAB":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRDEF_SATUI":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},"MRDEF_SUPPRESS":{"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}}}}}}}}
```

For each cui (accessible as the `_id`, and as a `cui` field), there are three attributes: `thesaurus`, `relations`, and `semtypes`. 

To retrieve a single cui, the following query may be used:

```bash 
curl -X GET https://localhost:9200/_search\?pretty -H "Content-Type: application/json" -d'{ "query": { "bool": { "should": [ { "match": { "cui": "YOURCUIGOESHERE" } } ] } }}'
```

