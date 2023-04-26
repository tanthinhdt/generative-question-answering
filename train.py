import evaluate
import argparse
import json
import torch
import math
from transformers import get_scheduler
from src.data.data_processor import DataProcessor
from models.t5.t5_small import T5Small
from models.bart.bart_base import BartBase
from tqdm.auto import tqdm


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs-path', '-c', type=str, required=True,
                        help='Path to the config file')
    return parser.parse_args()


class Trainer:
    def __init__(self, configs: dict) -> None:
        self.configs = configs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)

        self.tokenizer, self.model = self.get_tokenizer_and_model()
        self.model.to(self.device)
        self.data_processor = self.get_data_processor()
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           **self.configs['optimizer'])
        self.scheduler = self.get_scheduler()

        if self.configs['train']['from_checkpoint']:
            self.load_checkpoint()

    def get_tokenizer_and_model(self):
        model_dict = {
            't5-small': T5Small,
            'bart-base': BartBase
        }

        name = self.configs['name']
        pretrained = self.configs['pretrained']
        return model_dict[name](pretrained, **self.configs['model'])()

    def get_data_processor(self):
        dataset_dir = self.configs['data']['dir']
        tokenizer_info = {
            'tokenizer': self.tokenizer,
            'configs': self.configs['tokenizer']
        }
        return DataProcessor(tokenizer_info=tokenizer_info,
                             dataset_dir=dataset_dir)

    def get_scheduler(self):
        num_epochs = self.configs['train']['num_epochs']
        num_training_samples = self.configs['data']['num_training_samples']
        training_batch_size = self.configs['train']['batch_size']
        train_loader_length = math.ceil(
            num_training_samples / training_batch_size
        )
        num_training_steps = num_epochs * train_loader_length
        self.configs['train']['num_training_steps'] = num_training_steps
        scheduler = get_scheduler('linear',
                                  optimizer=self.optimizer,
                                  num_warmup_steps=0,
                                  num_training_steps=num_training_steps)
        return scheduler

    def train(self):
        num_epochs = self.configs['train']['num_epochs']
        num_trainining_steps = self.configs['train']['num_training_steps']
        num_logging_steps = self.configs['train']['num_logging_steps']
        num_saving_steps = self.configs['train']['num_saving_steps']
        batch_size = self.configs['train']['batch_size']
        train_loader = self.data_processor.get_train_loader(batch_size)

        self.model.train()
        progress_bar = tqdm(range(num_trainining_steps))
        steps = 0
        for _ in range(num_epochs):
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)
                if steps != 0 and steps % num_logging_steps == 0:
                    progress_bar.write(f'Loss: {loss.item()}')

                if steps != 0 and steps % num_saving_steps == 0:
                    self.save(steps)

    def evaluate(self):
        batch_size = self.configs['eval']['batch_size']
        eval_loader = self.data_processor.get_eval_loader(batch_size)
        metric = evaluate.load('rouge')

        self.model.eval()
        for batch in eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            inferences = self.tokenizer.decode(outputs[0],
                                               skip_special_tokens=True)
            answers = batch['answers']
            metric.add_batch(predictions=inferences,
                             references=answers)
        return metric.compute

    def from_checkpoint(self):
        model_checkpoint_path = self.configs['checpoints']['model']
        self.model.load_state_dict(torch.load(model_checkpoint_path))

        optimizer_checkpoint_path = self.configs['checpoints']['optimizer']
        self.optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))

        scheduler_checkpoint_path = self.configs['checpoints']['scheduler']
        self.scheduler.load_state_dict(torch.load(scheduler_checkpoint_path))

    def save(self, step: int):
        checkpoint_dir = self.configs['train']['checkpoint_dir']

        model_checkpoint_path = checkpoint_dir + f'model_{step}.pt'
        torch.save(self.model.state_dict(), model_checkpoint_path)

        optimizer_checkpoint_path = checkpoint_dir + f'optimizer_{step}.pt'
        torch.save(self.optimizer.state_dict(), optimizer_checkpoint_path)

        scheduler_checkpoint_path = checkpoint_dir + f'scheduler_{step}.pt'
        torch.save(self.scheduler.state_dict(), scheduler_checkpoint_path)


if __name__ == '__main__':
    args = get_args()
    configs_path = args.configs_path

    # for debugging
    # configs_path = 'configs/t5/t5_small.json'

    with open(configs_path, 'r') as f:
        configs = json.load(f)

    trainer = Trainer(configs)
    trainer.train()
