import os
import torch
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
from src.data.eli5 import ELI5
from src.utils import remove_redundant_spaces


class DataProcessor:
    def __init__(self, tokenizer_info: dict, dataset_dir: str = None):
        """
        Initialize the data processor.

        Parameters:
            tokenizer_info: dict
                The dictionary storing tokenizer and options for tokenization.
            dataset_dir: str, default=None
                The directory of the dataset.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)

        self.tokenizer = tokenizer_info['tokenizer']
        self.tokenizer_configs = tokenizer_info['configs']
        self.dataset_dir = dataset_dir
        self.dataset = self.get_dataset()

    def get_dataset(self):
        """
        Get the dataset from the directory.

        Returns:
            dict: The dataset.
        """
        dataset_dict = {
            'ELI5': ELI5
        }
        dataset_name = os.path.basename(self.dataset_dir)
        return dataset_dict[dataset_name](self.dataset_dir)()

    def get_train_set(self):
        """
        Return the training set processed into the input format of the model.

        Returns:
            datasets.IterableDataset: The training set.
        """
        train_set = self.dataset['train'].map(
            self.__process_sample,
            remove_columns=['question_id', 'question', 'ctxs', 'answers']
        )
        return train_set

    def get_train_loader(self, batch_size: int):
        """
        Return the training loader.

        Parameters:
            batch_size: int
                The batch size of the training loader.

        Returns:
            torch.utils.data.DataLoader: The training loader.
        """
        train_set = self.get_train_set()
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  collate_fn=DefaultDataCollator())
        return train_loader

    def get_eval_set(self):
        """
        Return the evaluation set processed into the input format of the model.

        Returns:
            datasets.IterableDataset: The evaluation set.
        """
        eval_set = self.dataset['eval'].map(
            self.__process_sample,
            remove_columns=['question_id', 'question', 'ctxs']
        )
        return eval_set

    def eval_collate_fn(self, batch: list):
        """
        Collate function for the evaluation set.

        Parameters:
            batch: list
                A list of samples.

        Returns:
            dict: The collated batch.
        """
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        answers_list = []
        for sample in batch:
            input_ids_list.append(sample['input_ids'])
            attention_mask_list.append(sample['attention_mask'])
            labels_list.append(sample['labels'])
            answers_list.append(sample['answers'])
        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list),
            'answers': answers_list
        }

    def get_eval_loader(self, batch_size: int):
        """
        Return the evaluation loader.

        Parameters:
            batch_size: int
                The batch size of the evaluation loader.

        Returns:
            torch.utils.data.DataLoader: The evaluation loader.
        """
        eval_set = self.get_eval_set()
        eval_loader = DataLoader(eval_set,
                                 batch_size=batch_size,
                                 collate_fn=self.eval_collate_fn)
        return eval_loader

    def __process_sample(self, sample: dict):
        """
        Process a sample into the input format of the model.

        Parameters:
            sample: dict
                A sample from the dataset.

        Returns:
            dict: The processed sample.
        """
        question = 'Question: '
        question += remove_redundant_spaces(sample['question'])
        question += ', Context: '
        question += ', '.join([remove_redundant_spaces(c)
                               for c in sample['ctxs']])

        answer = ', '.join([remove_redundant_spaces(a)
                           for a in sample['answers']])

        encoder_inputs = self.tokenizer(question, **self.tokenizer_configs)
        input_ids = encoder_inputs['input_ids']
        attention_mask = encoder_inputs['attention_mask']

        decoder_inputs = self.tokenizer(answer, **self.tokenizer_configs)
        labels = decoder_inputs['input_ids']

        labels[labels == self.tokenizer.pad_token_id] = -100

        sample = {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'labels': labels.squeeze(0)
        }

        return sample
