import argparse
from src import MyTrainer
from src import Inference


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs_path', '-c', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--question', '-q', type=str,
                        help='Question to be asked')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    configs_path = args.configs_path
    question = args.question

    with open(configs_path, 'r') as f:
        configs = json.load(f)

    if question:
        inference = Inference(configs)
        print(inference.infer(question))
    else:
        trainer = MyTrainer(configs)
        trainer.train()
