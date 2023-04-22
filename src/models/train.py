import evaluate
from transformers import Trainer, TrainingArguments
from models.bart.bart_base import BartBase
from models.t5.t5_small import T5Small
from src.data.data_processor import DataProcessor


class MyTrainer:
    def __init__(self, configs) -> None:
        self.configs = configs
        self.tokenizer, self.model = self.get_tokenizer_and_model()
        self.train_set, self.val_set = self.get_datasets(self.tokenizer)
        self.metric = evaluate.load('rouge')

    def get_tokenizer_and_model(self):
        model_dict = {
            'bart-base': BartBase,
            't5-small': T5Small
        }
        model_name = self.configs['model']['name']
        pretrained_model = self.configs['model']['pretrained']
        model_configs = self.configs['model'].get('configs', None)
        return model_dict[model_name](pretrained_model, model_configs)()

    def get_datasets(self, tokenizer):
        dataset_dir = self.configs['dataset']['dir']
        tokenizer_info = {
            'tokenizer': tokenizer,
            'configs': self.configs['tokenizer']
        }
        data_processor = DataProcessor(tokenizer_info=tokenizer_info,
                                       dataset_dir=dataset_dir)
        return data_processor.get_datasets()

    def train(self):
        train_args = TrainingArguments(**self.configs['train'])
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_set,
            eval_dataset=self.val_set,
            compute_metrics=self.compute_metrics
        )
        trainer.train()

    def compute_metrics(self, eval_pred):
        inference, answer = eval_pred
        rouge = self.metric.compute(predictions=inference,
                                    references=answer,
                                    rouge_type=['rouge_L'],
                                    use_aggregator=True)
        return {
            'Rouge-L': rouge['rougeL']
        }
