from datasets import load_dataset, DownloadConfig


class WikiSnippets:
    def __init__(self, option: str = 'wiki40b_en_100_0') -> None:
        self.option = option
        self.download_config = DownloadConfig(resume_download=True)

    def __call__(self):
        dataset = load_dataset('wiki_snippets', self.option, streaming=True)
        return dataset['train']
