import os
import json
import torch
from knowledge_hub import KnowledgeHub
from models.bart.bart_base import BartBase
from models.t5.t5_small import T5Small
from src.utils import remove_redundant_spaces


def get_args():
    """
    Get the arguments from the command line.

    Returns:
        argparse.Namespace: Arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs-path', '-c', type=str,
                        default='configs/inference.json',
                        help='Path to the config file')
    return parser.parse_args()


class Inference():
    def __init__(self, configs) -> None:
        """
        Initialize the inference.

        Parameters:
            configs: dict
                The configs of the inference.
        """
        self.configs = configs
        self.tokenizer, self.model = self.get_tokenizer_and_model()
        self.load_checkpoint()
        with open(self.configs['knowledge_hub'], 'r') as f:
            knowledge_hub_configs = json.load(f)
        self.knowledge_hub = KnowledgeHub(knowledge_hub_configs)

    def get_tokenizer_and_model(self):
        """
        Get the tokenizer and model defined in the config file.
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

    def load_checkpoint(self):
        """
        Load the checkpoint defined in the config file.
        """
        entry = self.configs['resume']['entry']
        step = self.configs['resume']['step']
        checkpoint_dir = self.configs['resume']['checkpoint_dir']

        model_checkpoint_path = os.path.join(checkpoint_dir, entry,
                                             f'model_{step}.pt')
        self.model.load_state_dict(torch.load(model_checkpoint_path))

    def process_question(self, question: str, support_documents: list = None):
        """
        Process the question and support documents into the input format
        of the model.

        Parameters:
            question: str
                The question to be answered.
            support_document: list
                The support documents for the question.
        """
        question = 'Answer this question: ' + remove_redundant_spaces(question)
        if support_documents:
            question += ', with the context: '
            question += ', '.join([remove_redundant_spaces(d)
                                   for d in support_documents])
        return self.tokenizer(question, **self.configs['tokenizer'])

    def infer(self, question: str):
        """
        Infer the answer to the question.

        Parameters:
            question: str
                The question to be answered.
        """
        top_k = self.configs['num_support_documents']
        support_documents = None
        if top_k > 0:
            support_documents = self.knowledge_hub.query(question, top_k)
        encoder_inputs = self.process_question(question, support_documents)
        output = self.model.generate(
            input_ids=encoder_inputs['input_ids'],
            attention_mask=encoder_inputs['attention_mask'],
            max_length=512,
            num_beams=2
        )
        inference = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return inference, support_documents


if __name__ == '__main__':
    args = get_args()
    configs_path = args.configs_path

    with open(configs_path, 'r') as f:
        configs = json.load(f)
    inference = Inference(configs)

    # for debugging
    # question = 'Why are different tiers (regular < mid < premium) of gas' prices almost always 10 cents different?'

    print('\nEnter q to quit')
    question = input('>>>>> Enter your question: ')
    while question != 'q':
        answer, support_documents = inference.infer(question)
        print('\n# Supporting documents:')
        print(*[f'{i + 1}. {d}' for i, d in enumerate(support_documents)],
              sep='\n')
        print('\n# Answer:\n' + answer)
        question = input('\n>>>>> Enter your question: ')
