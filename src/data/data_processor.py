import os
from src.data.eli5 import ELI5
from src.utils import remove_redundant_spaces


class DataProcessor:
    def __init__(self, tokenizer_info, dataset_dir: str = None):
        self.tokenizer = tokenizer_info['tokenizer']
        self.tokenizer_configs = tokenizer_info['configs']
        self.dataset_dir = dataset_dir

    def get_datasets(self):
        dataset_dict = {
            'ELI5': ELI5
        }
        dataset_name = os.path.basename(self.dataset_dir)
        dataset = dataset_dict[dataset_name](self.dataset_dir)()

        train_set = dataset['train'].map(
            self.__process_train_sample,
            remove_columns=dataset['train'].column_names
        )

        val_set = dataset['val'].map(
            self.__process_val_sample,
            remove_columns=dataset['val'].column_names
        )

        return train_set, val_set

    def __process_train_sample(self, sample: dict):
        question = remove_redundant_spaces(sample['question'])
        ctxs = ', '.join([remove_redundant_spaces(c) for c in sample['ctxs']])
        answer = ', '.join([remove_redundant_spaces(a)
                           for a in sample['answers']])

        question_plus = f'Answer this question: {question}'
        question_plus += f', with the context: {ctxs}'
        answer_plus = f'{answer}'

        return self.encode_sample(question_plus, answer_plus)

    def __process_val_sample(self, sample: dict):
        question = remove_redundant_spaces(sample['question'])
        ctxs = ', '.join([remove_redundant_spaces(i[0])
                         for i in sample['ctxs']])
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

        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'labels': labels.squeeze(0)
        }
