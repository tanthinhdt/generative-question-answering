import os
import torch
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
from src.data.eli5 import ELI5
from src.utils import remove_redundant_spaces


class DataProcessor:
    def __init__(self, tokenizer_info: dict, dataset_dir: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)

        self.tokenizer = tokenizer_info['tokenizer']
        self.tokenizer_configs = tokenizer_info['configs']
        self.dataset_dir = dataset_dir
        self.dataset = self.get_dataset()

    def get_dataset(self):
        dataset_dict = {
            'ELI5': ELI5
        }
        dataset_name = os.path.basename(self.dataset_dir)
        return dataset_dict[dataset_name](self.dataset_dir)()

    def get_train_set(self):
        train_set = self.dataset['train'].map(
            self.__process_sample,
            remove_columns=['question_id', 'question', 'ctxs', 'answers']
        )
        return train_set

    def get_train_loader(self, batch_size: int):
        train_set = self.get_train_set()
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  collate_fn=DefaultDataCollator())
        return train_loader

    def get_eval_set(self):
        eval_set = self.dataset['eval'].map(
            self.__process_sample,
            remove_columns=['question_id', 'question', 'ctxs']
        )
        return eval_set

    def eval_collate_fn(self, batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        answers_list = []
        for sample in batch:
            input_ids_list.append(sample['input_ids'])
            attention_mask_list.append(sample['attention_mask'])
            labels_list.append(sample['labels'])
            answers_list.append(sample['answers'])
        # input_ids_list = torch.stack(input_ids_list)
        # attention_mask_list = torch.stack(attention_mask_list)
        # labels_list = torch.stack(labels_list)
        # return input_ids_list, attention_mask_list, labels_list, answers_list
        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list),
            'answers': answers_list
        }

    def get_eval_loader(self, batch_size: int):
        eval_set = self.get_eval_set()
        eval_loader = DataLoader(eval_set,
                                 batch_size=batch_size,
                                 collate_fn=self.eval_collate_fn)
        return eval_loader

    def __process_sample(self, sample: dict):
        question = remove_redundant_spaces(sample['question'])
        ctxs = ', '.join([remove_redundant_spaces(c) for c in sample['ctxs']])
        answer = ', '.join([remove_redundant_spaces(a)
                           for a in sample['answers']])

        question_plus = f'Answer this question: {question}'
        question_plus += f', with the context: {ctxs}'
        answer_plus = f'{answer}'

        return self.encode_sample(question_plus, answer_plus)

    def encode(self, text: str):
        return self.tokenizer(text, **self.tokenizer_configs)

    def encode_sample(self, question: str, answer: str):
        encoder_inputs = self.encode(question)
        input_ids = encoder_inputs['input_ids']
        attention_mask = encoder_inputs['attention_mask']

        decoder_inputs = self.encode(answer)
        labels = decoder_inputs['input_ids']

        labels[labels == self.tokenizer.pad_token_id] = -100

        sample = {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'labels': labels.squeeze(0)
        }

        return sample
