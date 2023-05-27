import unittest
from lorem_text import lorem
from network import Transformer, T5Config
from transformers import T5Tokenizer
from datasets import Dataset


def create_dummy_text_data(n_questions: int, n_passages: int):
    data = []
    for _ in range(n_questions):
        question = lorem.sentence()
        answer = lorem.sentence()
        titles = [lorem.sentence() for _ in range(n_passages)]
        passages = [lorem.paragraph() for _ in range(n_passages)]
        data.append({
            'question': question,
            'answer': answer,
            'titles': titles,
            'passages': passages
        })
    return Dataset.from_list(data)


def create_data_loader(dataset: Dataset, prefixes: dict,
                       tokenizer: T5Tokenizer, tokenizer_configs: dict,
                       batch_size: int):
    def process_sample(sample):
        question = sample['question']
        answer = sample['answer']
        titles = sample['titles']
        passages = sample['passages']

        questions = []
        for title, passage in zip(titles, passages):
            question = f"{prefixes['question']} {question} "
            question += f"{prefixes['title']} {title} "
            question += f"{prefixes['context']} {passage}"
            questions.append(question)

        encoder_inputs = tokenizer.batch_encode_plus(
            questions,
            **tokenizer_configs
        )

        decoder_inputs = tokenizer.batch_encode_plus(
            [answer] * len(questions),
            **tokenizer_configs
        )

        # [n_passages, max_length]
        encoder_input_tokens = encoder_inputs['input_ids'].squeeze(0)
        decoder_input_tokens = decoder_inputs['input_ids'].squeeze(0)
        decoder_target_tokens = decoder_input_tokens[:, 1:]
        return {
            'encoder_input_tokens': encoder_input_tokens,
            'decoder_input_tokens': decoder_input_tokens,
            'decoder_target_tokens': decoder_target_tokens
        }

    dataset = dataset.map(process_sample).with_format('jax')
    return dataset.iter(batch_size=batch_size)


def test_fid_model():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    tokenizer_configs = {
        'max_length': 512,
        'padding': 'max_length',
        'truncation': True,
        'return_tensors': 'pt'
    }
    config = T5Config(vocab_size=tokenizer.vocab_size)
    fid_model = Transformer(config)

    n_questions = 50
    n_passages = 5
    prefixes = {
        'question': 'question: ',
        'title': 'title: ',
        'context': 'context: '
    }
    batch_size = 8

    dataset = create_dummy_text_data(n_questions, n_passages)
    for sample in dataset:
        assert 'question' in sample, 'Missing question'
        assert 'answer' in sample, 'Missing answer'
        assert 'titles' in sample, 'Missing titles'
        assert isinstance(sample['titles'], list), 'Titles is not a list'
        assert sample['titles'], 'Empty titles'
        assert 'passages' in sample, 'Missing passages'
        assert isinstance(sample['passages'], list), 'Passages is not a list'
        assert sample['passages'], 'Empty passages'

    data_loader = create_data_loader(dataset, prefixes,
                                     tokenizer, tokenizer_configs,
                                     batch_size)
    for batch in data_loader:
        encoder_input_tokens = batch['encoder_input_tokens']
        assert encoder_input_tokens.shape[0] == batch_size, (
            f'Expected batch size {batch_size}, '
            f'got {encoder_input_tokens.shape[0]}')
        assert encoder_input_tokens.shape[1] == n_passages, (
            f'Expected n_passages {n_passages}, '
            f'got {encoder_input_tokens.shape[1]}')
        assert encoder_input_tokens.shape[2] == tokenizer_configs['max_length'], (
            f'Expected max_length {tokenizer_configs["max_length"]}, '
            f'got {encoder_input_tokens.shape[2]}')

        decoder_input_tokens = batch['decoder_input_tokens']
        assert decoder_input_tokens.shape[0] == batch_size, (
            f'Expected batch size {batch_size}, '
            f'got {decoder_input_tokens.shape[0]}')
        assert decoder_input_tokens.shape[1] == n_passages, (
            f'Expected n_passages 1, '
            f'got {decoder_input_tokens.shape[1]}')
        assert decoder_input_tokens.shape[2] == tokenizer_configs['max_length'], (
            f'Expected max_length {tokenizer_configs["max_length"]}, '
            f'got {decoder_input_tokens.shape[2]}')

        decoder_target_tokens = batch['decoder_target_tokens']
        assert decoder_target_tokens.shape[0] == batch_size, (
            f'Expected batch size {batch_size}, '
            f'got {decoder_target_tokens.shape[0]}')
        assert decoder_target_tokens.shape[1] == n_passages, (
            f'Expected n_passages 1, '
            f'got {decoder_target_tokens.shape[1]}')
        assert decoder_target_tokens.shape[2] == tokenizer_configs['max_length'], (
            f'Expected max_length {tokenizer_configs["max_length"]}, '
            f'got {decoder_target_tokens.shape[2]}')

        output = fid_model(encoder_input_tokens,
                           decoder_input_tokens,
                           decoder_target_tokens)
        print(output)
        break


if __name__ == '__main__':
    unittest.main()
