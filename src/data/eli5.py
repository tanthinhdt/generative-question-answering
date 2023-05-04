import os
from datasets import load_dataset


class ELI5:
    def __init__(self, dataset_dir: str):
        """
        Initialize the dataset.

        Parameters:
            dataset_dir: str
                The directory of the dataset.
        """
        self.data_files = {
            'train': os.path.join(dataset_dir, 'train.jsonl'),
            'eval': os.path.join(dataset_dir, 'val.jsonl')
        }

    def __call__(self):
        """
        Return the streaming dataset.

        Returns:
            dict: The dataset.
        """
        dataset = load_dataset('json', data_files=self.data_files,
                               streaming=True, keep_in_memory=False)
        dataset['eval'] = dataset['eval'].map(self.__tailor_val_set)
        return dataset

    def __tailor_val_set(self, sample: dict):
        """
        Remove scores from the contexts of validation set.

        Parameters:
            sample: dict
                The sample.

        Returns:
            dict: The sample.
        """
        new_ctxs = []
        for ctx in sample['ctxs']:
            new_ctxs.append(ctx[0])
        sample['ctxs'] = new_ctxs
        return sample
