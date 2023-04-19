import os
from datasets import load_dataset


class ELI5:
    def __init__(self, dataset_dir):
        self.data_files = {
            'train': os.path.join(dataset_dir, 'train.jsonl'),
            'test': os.path.join(dataset_dir, 'val.jsonl')
        }

    def __call__(self):
        return load_dataset('json', data_files=self.data_files, streaming=True)
