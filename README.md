# Generative Question Answering
In this project, I trained some well-known architectures for generative question answering and evaluated these models on ROUGE-L metric. I also created a database to provide supporting documents for these model when inferring.

# Installing requirements
## Esentials Libraries
```shell
pip install -r requirements
```

## Postgresql
On Ubuntu:
```shell
# Create the file repository configuration:
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Import the repository signing key:
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

# Update the package lists:
sudo apt-get update

# Install PostgreSQL and supporting libraries.
# In this project, I'm using version 12, so I specify the installation as follow
sudo apt-get -y install postgresql-12 postgresql-server-dev-12 libpq-dev
```

Before inference or inserting knowledge to the database, remember to start the Postgresql first by using these commands:

```shell
# Check if Postgresql has started yet 
sudo service postgresql status

# If the status is 'down', we have turn it on
sudo service postgresql start
```


## pgvector
pgvector is a powerful tool that helps storing vectors in Postgresql databases and searching vectors computing similarity with the given query. pqvector supports methods such as:

* Exact and approximate nearest neighbor search
* L2 distance, inner product, and cosine distance

To install on Ubuntu, please see this [installation guide](https://github.com/pgvector/pgvector). If you run into problems with `make` command, this [discussion](https://askubuntu.com/questions/1095168/command-not-found-cc-make-error-127) may help.



# Training
To train a model defined in this project, you have to set up parameters for training in a config file. You can find the sample of this config file in [configs](/configs/) directory. Finally, you pass the path to the config file to this command to start training.
```shell
python3 train.py -c <path-to-config-file>
```

For example:
```
python3 train.py -c configs/t5/t5_small.json
```

# Inference
Before inferring, remember to start Postgresql service first (see [Postgresql section](#postgresql)). Note that, configurations for inference must also be set up in a config file and its path has to be passed down to this command. You can find the sample of the file [here](/configs/inference.json).

```shell
python3 infer.py
```

# Knowledge Hub
Before interacting with the knowledge hub, remember to start Postgresql service first (see [Postgresql section](#postgresql)). Note that, configurations for knowledge hub must also be set up in a config file and its path has to be passed down to this command. You can find the sample of the file [here](/configs/knowledge_hub.json).
## Inserting knowledge
```shell
python3 knowledge_hub.py -n <number-of-samples-to-be-inserted>
```

For example:
```shell
# Inserting 1,000,000 samples
python3 knowledge_hub.py -n 1000000
```

## Storing and searching vectors based on similarity
To see how I use `pgvector` and `Postgresql` to store and search vectors by using some common methods to compute similarity, checkout [pqvector guide](https://github.com/pgvector/pgvector) and [here](/knowledge_hub.py).

# Results

