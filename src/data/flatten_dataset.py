import polars as pl
from tqdm import tqdm
from datasets import load_dataset


def flatten_train_set(train_set):
    df = pl.DataFrame(
        schema={'id': str, 'question': str, 'answer': str, 'context': str})

    for itr in tqdm(train_set, leave=True):
        df1 = pl.DataFrame({
            'id': [itr['question_id']],
            'question': [itr['question']],
        })

        answers = itr['answers'] if itr['answers'] else ['']
        df2 = pl.DataFrame({
            'id': [itr['question_id']] * len(answers),
            'answer': answers
        })

        contexts = itr['ctxs'] if itr['ctxs'] else ['']
        df3 = pl.DataFrame({
            'id': [itr['question_id']] * len(contexts),
            'context': contexts
        })

        sample = (df1
                  .join(df2, on='id', how='outer')
                  .join(df3, on='id', how='outer'))

        df = pl.concat([df, sample])

    return df


data_files = {
    'train': '../../data/ELI5/train.jsonl',
    'test': '../../data/ELI5/val.jsonl'
}

dataset = load_dataset('json', data_files=data_files, streaming=True)

train_path = '../../data/final/ELI5/train.json'
train_df = flatten_train_set(dataset['train'])
train_df.write_json(train_path)

print('Number of samples in new train set:', pl.count(train_df['id']))

# test_df = create_df(dataset['test'])
# test_df.write_json('../../data/final/ELI5/val.json')
# print('Number of samples in new test set:', pl.count(test_df['id']))
