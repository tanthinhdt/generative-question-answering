import psycopg2
import json
import torch
import math
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.wiki_snippets import WikiSnippets
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs-path', '-c', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--n-samples', '-n', type=int,
                        default=5,
                        help='Number of samples to be inserted if mode is insert knowledge')
    return parser.parse_args()


class KnowledgeHub:
    def __init__(self, configs: dict) -> None:
        self.configs = configs
        self.connection_configs = configs['connection']
        with open(self.connection_configs['configs_path'], 'r') as f:
            self.connection_configs['configs'] = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(configs['pretrained'])
        self.model = AutoModel.from_pretrained(configs['pretrained'])

    def query(self, question: str, top_k: int = 3):
        src = self.configs['src']
        question_embedding = str(self.encode(question))
        metric_dict = {
            'cosine_similarity': f"1 - (embedding <=> '{question_embedding}')",
            'l2': f"embedding <-> '{question_embedding}'",
            'inner_product': f"(embedding <#> '{question_embedding}') * -1"
        }
        metric = metric_dict[self.configs['metric']]

        with psycopg2.connect(**self.connection_configs['configs']) as conn:
            cur = conn.cursor()

            query_script = f'SELECT passage FROM {src} '
            query_script += f'ORDER BY {metric} DESC LIMIT {top_k}'
            cur.execute(query_script)
            texts = cur.fetchall()

            cur.close()
        conn.close()

        return [text[0] for text in texts]

    def insert(self, n_samples: int):
        datasets_dict = {
            'wiki40b': WikiSnippets('wiki40b_en_100_0'),
            'wikipedia': WikiSnippets('wikipedia_en_100_0')
        }
        src = self.configs['src']
        batch_size = self.configs['batch_size']
        n_batches = math.ceil(n_samples / batch_size)
        dataset = datasets_dict[src]()
        dataset = dataset.map(
            remove_columns=['_id', 'start_paragraph', 'end_paragraph',
                            'start_character', 'end_character',
                            'article_title', 'section_title']
        )
        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 collate_fn=self.collate_fn)

        with psycopg2.connect(**self.connection_configs['configs']) as conn:
            cur = conn.cursor()

            count = 0
            progress_bar = tqdm(range(n_batches), desc='Inserting knowledge',
                                total=n_batches, unit='batch', leave=False)
            for batch in data_loader:
                ids = batch['ids']
                texts = batch['passage_texts']
                embeddings = self.encode(texts)

                if self.insert_batch(cur, src, ids, texts, embeddings):
                    count += 1
                    progress_bar.update(1)

                if count == n_samples:
                    break

            progress_bar.close()
            print(f'Inserted {count} samples')
            cur.close()

    def insert_batch(self, cur, src: str,
                     ids: str, texts: str, embeddings: list):
        state = False
        for id, text, embedding in zip(ids, texts, embeddings):
            embedding = str(embedding)
            try:
                cur.execute(
                    f'INSERT INTO {src}' + ' VALUES (%s, %s, %s)',
                    (id, text, embedding)
                )
                state = True
            except psycopg2.errors.UniqueViolation:
                cur.execute('ROLLBACK')
        return state

    def insert_sample(self, cur, src: str, id: str, text: str, embedding: str):
        state = False
        try:
            cur.execute(
                f'INSERT INTO {src}' + ' VALUES (%s, %s, %s)',
                (id, text, embedding)
            )
            state = True
        except psycopg2.errors.UniqueViolation:
            cur.execute('ROLLBACK')
        return state

    def encode(self, text: str):
        encoded_input = self.tokenizer(text, padding=True, truncation=True,
                                       return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        embeddings = self.mean_pooling(model_output,
                                       encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.squeeze(0).tolist()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (attention_mask
                               .unsqueeze(-1)
                               .expand(token_embeddings.size())
                               .float())
        result = torch.sum(token_embeddings * input_mask_expanded, 1)
        result /= torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return result

    def collate_fn(self, batch):
        id_list = []
        passage_text_list = []
        for sample in batch:
            id_list.append(str(sample['datasets_id']) + sample['wiki_id'])
            passage_text_list.append(sample['passage_text'])
        return {
            'ids': id_list,
            'passage_texts': passage_text_list
        }


if __name__ == '__main__':
    args = get_args()
    configs_path = args.configs_path
    n_samples = args.n_samples

    # for debugging
    # configs_path = 'configs/knowledge_hub.json'
    # n_samples = 5

    with open(configs_path, 'r') as f:
        configs = json.load(f)

    knowledge_hub = KnowledgeHub(configs)
    knowledge_hub.insert(n_samples)
