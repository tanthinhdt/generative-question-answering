from models.bart.bart_base import BartBase
from models.t5.t5_small import T5Small
from src.data.data_processor import DataProcessor


class Inference():
    def __init__(self, configs) -> None:
        self.configs = configs
        self.model, self.tokenizer = self.get_tokenizer_and_model()
        tokenizer_info = {
            'tokenizer': self.tokenizer,
            'configs': self.configs['tokenizers']
        }
        self.data_processor = DataProcessor(tokenizer_info=tokenizer_info)

    def get_tokenizer_and_model(self):
        model_dict = {
            'bart_base': BartBase,
            't5_small': T5Small
        }
        model_name = self.configs['model']['model_name']
        pretrained_model = self.configs['inference']['pretrained_model']
        model_configs = self.configs['model']['model_configs']
        return model_dict[model_name](pretrained_model, model_configs)()

    def infer(self, question):
        encoder_inputs = self.data_processor.encode(question, 'encoder')
        input_ids = encoder_inputs['input_ids']
        attetion_mask = encoder_inputs['attention_mask']
        output = self.model.generate(input_ids=input_ids,
                                     attetion_mask=attetion_mask,
                                     max_length=512,
                                     num_beams=2)
        inference = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return inference
