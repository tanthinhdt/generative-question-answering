from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration


class T5Small:
    def __init__(self, pretrained: str = 't5-small', **configs):
        """
        Initialize the tokenizer and model.

        Parameters:
            pretrained: str, default='t5-small'
                The name of the pretrained model.
            configs: dict
                The configurations for the model.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained)

        # The decoder_start_token_id is set to the id of the pad token.
        decoder_start_token_id = (self.tokenizer
                                  .convert_tokens_to_ids(['<pad>'])[0])
        if not configs:
            configs = dict()
        configs['decoder_start_token_id'] = decoder_start_token_id
        self.configs = T5Config.from_pretrained(pretrained, **configs)
        self.model = T5ForConditionalGeneration(self.configs)
        self.model.from_pretrained(pretrained)

    def __call__(self):
        """
        Return the tokenizer and the model.

        Returns:
            transformers.T5Tokenizer: The tokenizer,
            transformers.T5ForConditionalGeneration: The model.
        """
        return self.tokenizer, self.model

    def get_tokenizer(self):
        """
        Return the tokenizer used by the model.

        Returns:
            transformers.T5Tokenizer: The tokenizer.
        """
        return self.tokenizer

    def get_model(self):
        """
        Return the model.

        Returns:
            transformers.T5ForConditionalGeneration: The model.
        """
        return self.model
