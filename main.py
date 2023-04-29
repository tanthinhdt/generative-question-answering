import argparse
import json

from src import Inference, Trainer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', '-m', type=str, required=True,
    #                     default='train',
    #                     help='Mode to run (train, infer, insert knowledge)')
    parser.add_argument('--configs-path', '-c', type=str, required=True,
                        default='configs/t5/t5_small.json',
                        help='Path to the config file')
    parser.add_argument('--question', '-q', type=str,
                        default=None,
                        help='Question to be asked if mode is infer')
    # parser.add_argument('--top-k', '-k', type=int,
    #                     default=3,
    #                     help='Number of supporting documents to be retrieved if mode is infer')
    # parser.add_argument('--n-samples', '-n', type=int,
    #                     default=5,
    #                     help='Number of samples to be inserted if mode is insert knowledge')
    return parser.parse_args()


if __name__ == '__main__':
    # args = get_args()

    # mode = args.mode
    # configs_path = args.configs_path
    # question = args.question
    # src = args.src
    # n_samples = args.n_samples

    # for debugging
    # mode = 'train'
    configs_path = 'configs/t5/t5_small.json'
    question = None
    # src = 'wiki40b'
    # n_samples = 5

    with open(configs_path, 'r') as f:
        configs = json.load(f)

    if question:
        inference = Inference(configs)
        print(inference.infer(question))
    else:
        trainer = Trainer(configs)
        trainer.train()
