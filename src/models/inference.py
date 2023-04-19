import argparse
from models.bart.bart_base import BartBase
from models.t5.t5_small import T5Small


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', '-c', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--question', '-q', type=str, required=True,
                        help='Question to be asked')
    return parser.parse_args()


class Inference():
    def __init__(self, configs) -> None:
        self.configs = configs
        self.model = self.get_model()

    def get_tokenizer_and_model(self):
        model_dict = {
            'bart_base': BartBase,
            't5_small': T5Small
        }
        model_name = self.configs['model']['model_name']
        pretrained_model = self.configs['inference']['pretrained_model']
        model_configs = self.configs['model']['model_configs']
        return model_dict[model_name](pretrained_model, model_configs)

    def infer(self, question):
        return self.model.generate(question)


if __name__ == '__main__':
    args = get_args()

    configs = args.configs
    question = args.question

    inference = Inference(configs)
    print(inference.infer(question))
