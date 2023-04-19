import os
from src.data.eli5 import ELI5
from src.utils import preprocess


class Dataset:
    def __init__(self, dataset_dir, tokenizer, **tokenizer_configs):
        self.dataset_dir = dataset_dir

        self.tokenizer = tokenizer
        if not tokenizer_configs:
            tokenizer_configs = {
                'max_length': 384,
                'truncation': 'only_second',
                'stride': 128,
                'return_overflowing_tokens': True,
                'reurn_offsets_mapping': True,
                'padding': 'max_length'
            }
        self.tokenizer_configs = tokenizer_configs

    def get_datasets(self):
        dataset_dict = {
            'ELI5': ELI5
        }
        dataset_name = os.path.basename(self.dataset_dir)
        dataset = dataset_dict[dataset_name](self.dataset_dir)()

        train_set = dataset['train'].map(
            self.__preprocess_train_set,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        test_set = dataset['test'].map(
            self.__preprocess_test_set,
            batched=True,
            remove_columns=dataset['test'].column_names
        )
        
        return train_set, test_set

    def __preprocess_train_set(self, samples):
        questions = [preprocess(question)
                     for question in samples['question']]
        inputs = self.tokenizer(
            questions,
            samples['context'],
            **self.tokenizer_configs
        )
        return inputs

    def __preprocess_test_set(self, samples):
        questions = [preprocess(question)
                     for question in samples['question']]
        inputs = self.tokenizer(
            questions,
            samples['context'],
            **self.tokenizer_configs
        )
        return inputs
