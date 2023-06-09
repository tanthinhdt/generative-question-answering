import evaluate
import argparse
import json
import torch
import math
import os
from transformers import get_scheduler
from src.data.data_processor import DataProcessor
from models.t5.t5_small import T5Small
from models.bart.bart_base import BartBase
from tqdm.auto import tqdm


def get_args() -> argparse.Namespace:
    """
    Get the arguments from the command line.

    Returns:
        argparse.Namespace: The arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs-path', '-c', type=str, required=True,
                        help='Path to the config file')
    return parser.parse_args()


class Trainer:
    def __init__(self, configs: dict) -> None:
        """
        Initialize the trainer.

        Parameters:
            configs: dict
                The configurations for the trainer.
        """
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

        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'rougeL': [],
            'rougeLsum': []
        }

        self.saved_checkpoints = {
            'model': [],
            'optimizer': [],
            'scheduler': []
        }

    def get_tokenizer_and_model(self):
        """
        Return the tokenizer and model defined in the config file.

        Returns:
            transformers.PreTrainedTokenizer: The tokenizer,
            transformers.PreTrainedModel: The model.
        """
        model_dict = {
            't5-small': T5Small,
            'bart-base': BartBase
        }

        name = self.configs['name']
        pretrained = self.configs['pretrained']
        model_configs = self.configs.get('model', None)
        model_configs = model_configs if model_configs else dict()
        return model_dict[name](pretrained, **model_configs)()

    def get_data_processor(self):
        """
        Create and return a data processor for the dataset.

        Returns:
            DataProcessor: A data processor for the dataset.
        """
        dataset_dir = self.configs['data']['dir']
        tokenizer_info = {
            'tokenizer': self.tokenizer,
            'configs': self.configs['tokenizer']
        }
        return DataProcessor(tokenizer_info=tokenizer_info,
                             dataset_dir=dataset_dir)

    def get_scheduler(self):
        """
        Create and return a scheduler for the optimizer.

        Returns:
            torch.optim.lr_scheduler: A scheduler for the optimizer.
        """
        num_epochs = self.configs['train']['num_epochs']
        num_training_samples = self.configs['data']['num_training_samples']
        training_batch_size = self.configs['train']['batch_size']
        warmup_ratio = self.configs['scheduler']['warmup_ratio']
        train_loader_length = math.ceil(
            num_training_samples / training_batch_size
        )
        num_training_steps = num_epochs * train_loader_length
        num_warmpup_steps = int(num_training_steps * warmup_ratio)
        self.configs['train']['num_training_steps'] = num_training_steps
        scheduler = get_scheduler('linear',
                                  optimizer=self.optimizer,
                                  num_warmup_steps=num_warmpup_steps,
                                  num_training_steps=num_training_steps)
        return scheduler

    def train(self):
        """
        Train the model.
        """
        num_epochs = self.configs['train']['num_epochs']
        num_trainining_steps = self.configs['train']['num_training_steps']
        num_logging_steps = self.configs['train']['num_logging_steps']
        num_saving_steps = self.configs['train']['num_saving_steps']
        num_eval_steps = self.configs['train']['num_eval_steps']
        batch_size = self.configs['train']['batch_size']
        train_loader = self.data_processor.get_train_loader(batch_size)

        self.model.train()

        progress_bar = tqdm(range(num_trainining_steps),
                            total=num_trainining_steps,
                            desc='Training', unit='step', leave=False)
        start = 0
        if self.configs['train']['from_checkpoint']:
            start = self.configs['resume']['step']
        steps = start
        for _ in range(num_epochs):
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if steps != start and steps % num_logging_steps == 0:
                    message = f'Logging at {steps} >> Loss: {loss.item()}'
                    progress_bar.write(message)

                if steps != start and steps % num_saving_steps == 0:
                    self.save(steps)

                if steps != start and steps % num_eval_steps == 0:
                    self.history['train_loss'].append(loss.item())
                    eval_results = self.evaluate()
                    message = f'Evaluation at {steps} >> '
                    message += ', '.join(
                        [f'{m}: {v}' for m, v in eval_results.items()])
                    progress_bar.write(message)

                progress_bar.update(1)
                steps += 1
        progress_bar.close()

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        Metrics include: ROUGE-L, ROUGE-Lsum, Loss

        Returns:
            dict: A dictionary of evaluation results.
        """
        num_eval_samples = self.configs['data']['num_eval_samples']
        batch_size = self.configs['eval']['batch_size']
        eval_loader = self.data_processor.get_eval_loader(batch_size)
        metric = evaluate.load('rouge')

        self.model.eval()
        loss = 0
        for batch in eval_loader:
            batch = {k: v.to(self.device) if k != 'answers' else v
                     for k, v in batch.items()}
            with torch.no_grad():
                loss += self.model(input_ids=batch['input_ids'],
                                   attention_mask=batch['attention_mask'],
                                   labels=batch['labels']).loss.item()
                outputs = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    num_beams=2, max_length=32
                )

            inferences = self.tokenizer.batch_decode(outputs,
                                                     skip_special_tokens=True)
            metric.add_batch(predictions=inferences,
                             references=batch['answers'])
        eval_results = metric.compute(rouge_types=['rougeL', 'rougeLsum'])
        eval_results['loss'] = loss / math.ceil(num_eval_samples / batch_size)

        self.history['eval_loss'].append(eval_results['loss'])
        self.history['rougeL'].append(eval_results['rougeL'])
        self.history['rougeLsum'].append(eval_results['rougeLsum'])

        return eval_results

    def load_checkpoint(self):
        """
        Load the checkpoint defined in the config file.
        """
        entry = self.configs['resume']['entry']
        step = self.configs['resume']['step']
        checkpoint_dir = self.configs['train']['checkpoint_dir']

        model_checkpoint_path = os.path.join(checkpoint_dir, entry,
                                             f'model_{step}.pt')
        self.model.load_state_dict(torch.load(model_checkpoint_path))

        optimizer_checkpoint_path = os.path.join(checkpoint_dir, entry,
                                                 f'optimizer_{step}.pt')
        self.optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))

        scheduler_checkpoint_path = os.path.join(checkpoint_dir, entry,
                                                 f'scheduler_{step}.pt')
        self.scheduler.load_state_dict(torch.load(scheduler_checkpoint_path))

    def save(self, step: int):
        """
        Save the model, optimizer, scheduler states and training history.

        Parameters:
            step: int
                The current step.
        """
        entry = self.configs['entry']
        checkpoint_dir = os.path.join(self.configs['train']['checkpoint_dir'],
                                      entry)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        log_dir = os.path.join(self.configs['train']['log_dir'], entry)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        n_checkpoints = self.configs['train']['num_checkpoints']

        model_checkpoint = f'model_{step}.pt'
        model_checkpoint_path = os.path.join(checkpoint_dir, model_checkpoint)
        torch.save(self.model.state_dict(), model_checkpoint_path)
        self.saved_checkpoints['model'].append(model_checkpoint)
        if len(self.saved_checkpoints['model']) > n_checkpoints:
            os.remove(os.path.join(checkpoint_dir,
                                   self.saved_checkpoints['model'].pop(0)))

        optimizer_checkpoint = f'optimizer_{step}.pt'
        optimizer_checkpoint_path = os.path.join(checkpoint_dir,
                                                 optimizer_checkpoint)
        torch.save(self.optimizer.state_dict(), optimizer_checkpoint_path)
        self.saved_checkpoints['optimizer'].append(optimizer_checkpoint)
        if len(self.saved_checkpoints['optimizer']) > n_checkpoints:
            os.remove(os.path.join(checkpoint_dir,
                                   self.saved_checkpoints['optimizer'].pop(0)))

        scheduler_checkpoint = f'scheduler_{step}.pt'
        scheduler_checkpoint_path = os.path.join(checkpoint_dir,
                                                 scheduler_checkpoint)
        torch.save(self.scheduler.state_dict(), scheduler_checkpoint_path)
        self.saved_checkpoints['scheduler'].append(scheduler_checkpoint)
        if len(self.saved_checkpoints['scheduler']) > n_checkpoints:
            os.remove(os.path.join(checkpoint_dir,
                                   self.saved_checkpoints['scheduler'].pop(0)))

        log_path = checkpoint_dir + f'/{entry}.json'
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=4)


if __name__ == '__main__':
    args = get_args()
    configs_path = args.configs_path

    # for debugging
    # configs_path = 'configs/t5/t5_small.json'

    with open(configs_path, 'r') as f:
        configs = json.load(f)

    trainer = Trainer(configs)
    trainer.train()
