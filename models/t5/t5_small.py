from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration


class T5Small:
    def __init__(self, pretrained_model="t5-small", configs=None):
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
        self.configs = T5Config(**configs) if configs else T5Config()
        self.model = T5ForConditionalGeneration(self.configs)
        self.model.from_pretrained(pretrained_model)

    def __call__(self):
        return self.tokenizer, self.model
