from transformers import BartTokenizer, BartConfig
from transformers import BartForConditionalGeneration


class BartBase:
    def __init__(self, pretrained_model="facebook/bart-base", configs=None):
        self.tokenizer = BartTokenizer.from_pretrained(pretrained_model)

        decoder_start_token_id = (self.tokenizer
                                  .convert_tokens_to_ids(['<pad>'])[0])
        if not configs:
            configs = dict()
        configs['decoder_start_token_id'] = decoder_start_token_id
        self.configs = BartConfig.from_pretrained(pretrained_model, **configs)
        self.model = BartForConditionalGeneration(self.configs)
        self.model.from_pretrained(pretrained_model)

    def __call__(self):
        return self.tokenizer, self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
