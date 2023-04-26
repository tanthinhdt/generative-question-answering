from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration


class T5Small:
    def __init__(self, pretrained: str = 't5-small',
                 **configs):
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained)

        decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(['<pad>'])[0]
        if not configs:
            configs = dict({
                'decoder_start_token_id': decoder_start_token_id
            })
        self.configs = T5Config(**configs)
        self.model = T5ForConditionalGeneration(self.configs)
        self.model.from_pretrained(pretrained)

    def __call__(self):
        return self.tokenizer, self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
