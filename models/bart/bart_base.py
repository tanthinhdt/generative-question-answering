from transformers import BartTokenizer, BartConfig
from transformers import BartForConditionalGeneration


class BartBase:
    def __init__(self, pretrained: str = "facebook/bart-base", **configs):
        """
        Initialize the tokenizer and model.

        Parameters:
            pretrained: str, default='facebook/bart-base'
                The name of the pretrained model.
            configs: dict
                The configurations for the model.
        """
        self.tokenizer = BartTokenizer.from_pretrained(pretrained)

        # The decoder_start_token_id is set to the id of the pad token.
        decoder_start_token_id = (self.tokenizer
                                  .convert_tokens_to_ids(['<pad>'])[0])
        if not configs:
            configs = dict()
        configs['decoder_start_token_id'] = decoder_start_token_id
        self.configs = BartConfig.from_pretrained(pretrained, **configs)
        self.model = BartForConditionalGeneration(self.configs)
        self.model.from_pretrained(pretrained)

    def __call__(self):
        """
        Return the tokenizer and the model.

        Returns:
            transformers.BartTokenizer: The tokenizer,
            transformers.BartForConditionalGeneration: The model.
        """
        return self.tokenizer, self.model

    def get_tokenizer(self):
        """
        Return the tokenizer used by the model.

        Returns:
            transformers.BartTokenizer: The tokenizer.
        """
        return self.tokenizer

    def get_model(self):
        """
        Return the model.

        Returns:
            transformers.BartForConditionalGeneration: The model.
        """
        return self.model
