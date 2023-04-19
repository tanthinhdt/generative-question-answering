from transformers import BartTokenizer, BartConfig, BartForQuestionAnswering


class BartBase:
    def __init__(self, pretrained_model="facebook/bart-base", configs=None):
        self.tokenizer = BartTokenizer.from_pretrained(pretrained_model)
        self.configs = BartConfig(**configs) if configs else None
        self.model = BartForQuestionAnswering(self.configs)
        self.model.from_pretrained(pretrained_model)

    def __call__(self):
        return self.tokenizer, self.model
