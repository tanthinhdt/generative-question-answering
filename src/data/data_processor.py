import os
from src.data.eli5 import ELI5
from src.utils import remove_redundant_spaces
from transformers import DefaultDataCollator
from torch.utils.data import DataLoader


class DataProcessor:
    def __init__(self, tokenizer_info: dict, dataset_dir: str = None):
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
        eval_set = self.dataset['val'].map(
            self.__process_sample,
            remove_columns=['question_id', 'question', 'ctxs']
        )
        return eval_set

    def get_eval_loader(self, batch_size: int):
        eval_set = self.get_eval_set()
        eval_loader = DataLoader(eval_set,
                                 batch_size=batch_size,
                                 collate_fn=DefaultDataCollator())
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
        decoder_input_ids = decoder_inputs['input_ids']

        decoder_input_ids[decoder_input_ids ==
                          self.tokenizer.pad_token_id] = -100

        sample = {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'decoder_input_ids': decoder_input_ids.squeeze(0)
        }

        return sample
