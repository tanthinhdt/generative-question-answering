# Configurations
## For Knowledge Hub
Configurations are include in two files:

* [connection.json](/configs/connection.json) contains settings for connection to `Postgresql` database.
* [knowledge_hub.json](/configs/knowledge_hub.json) contains settings for the knowledge hub such as embedder and retrieval model, metric for searching, batch size for insertion,...

## For Inference
Configurations about model, tokenizer for inferring process are stored in [inference.json](/configs/inference.json)

## For models
Separated configuration files are placed in folders named according to the family of the model. These following models are supported in this project:

* [T5](/configs/t5/)
  * [T5-Small](/configs/t5/t5_small.json)
* [BART](/configs/bart/)
  * [BART-base](/configs/bart/bart_base.json)
