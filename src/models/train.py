import argparse
import json
import sys
sys.path.append('D:\\Projects\\generative-question-answering\\')
from models.bart.bart_base import BartBase
from models.t5.t5_small import T5Small
from src.data.dataset import Dataset
from transformers import TrainingArguments, Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs_path', '-c', type=str, required=True,
                        help='Path to the config file')
    return parser.parse_args()


class MyTrainer:
    def __init__(self, configs) -> None:
        self.configs = configs
        self.tokenizer, self.model = self.get_tokenizer_and_model()
        self.train_set, self.val_set = self.get_dataloaders(self.tokenizer)

    def get_tokenizer_and_model(self):
        model_dict = {
            'bart_base': BartBase,
            't5_small': T5Small
        }
        model_name = self.configs['model']['name']
        model_configs = self.configs['model']['configs']
        return model_dict[model_name](model_configs)

    def get_dataloaders(self, tokenizer):
        dataset_dir = self.configs['dataset']['dir']
        tokenizer_configs = self.configs['dataset']['tokenizer_configs']
        dataset = Dataset(dataset_dir, tokenizer, **tokenizer_configs)
        return dataset.get_dataloaders()

    def train(self):
        train_args = TrainingArguments(**self.configs['train'])
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_set,
            eval_dataset=self.val_set,
            tokenizer=self.tokenizer
        )
        trainer.train()


if __name__ == '__main__':
    args = get_args()

    configs_path = args.configs_path

    with open(configs_path, 'r') as f:
        configs = json.load(f)

    trainer = MyTrainer(configs)
    trainer.train()
