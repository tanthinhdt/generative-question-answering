from models.bart.bart_base import BartBase
from models.t5.t5_small import T5Small


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
