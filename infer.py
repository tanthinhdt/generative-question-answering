from transformers import BartConfig, T5Config
from transformers import AutoModel, AutoTokenizer
from src.data.data_processor import DataProcessor
from knowledge_hub import KnowledgeHub


class Inference():
    def __init__(self, configs) -> None:
        self.configs = configs
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        tokenizer_info = {
            'tokenizer': self.tokenizer,
            'configs': self.configs['tokenizers']
        }
        self.data_processor = DataProcessor(tokenizer_info=tokenizer_info)

    def get_tokenizer(self):
        pretrained = self.configs['pretrained']
        return AutoTokenizer.from_pretrained(pretrained)

    def get_model(self):
        model_family_dict = {
            'bart': BartConfig,
            't5': T5Config
        }
        pretrained = self.configs['pretrained']
        decoder_start_token_id = (self.tokenizer
                                  .convert_tokens_to_ids(['<pad>'])[0])
        model_configs = model_family_dict[self.configs['type']](
            decoder_start_token_id=decoder_start_token_id,
            **self.configs['model']
        )
        model = AutoModel.from_config(model_configs)
        return model.from_pretrained(pretrained)

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
