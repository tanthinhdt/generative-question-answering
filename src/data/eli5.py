import os
from datasets import load_dataset


class ELI5:
    def __init__(self, dataset_dir):
        self.data_files = {
            'train': os.path.join(dataset_dir, 'train.jsonl'),
            'eval': os.path.join(dataset_dir, 'val.jsonl')
        }

    def __call__(self):
        dataset = load_dataset('json', data_files=self.data_files,
                               streaming=True, keep_in_memory=False)
        dataset['eval'] = dataset['eval'].map(self.__tailor_val_set)
        return dataset

    def __tailor_val_set(self, sample: dict):
        new_ctxs = []
        for ctx in sample['ctxs']:
            new_ctxs.append(ctx[0])
        sample['ctxs'] = new_ctxs
        return sample
