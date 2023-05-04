from datasets import load_dataset, DownloadConfig


class WikiSnippets:
    def __init__(self, option: str = 'wiki40b_en_100_0') -> None:
        """
        Initialize the dataset.

        Parameters:
            option: str, default='wiki40b_en_100_0'
                The option of the dataset.
        """
        self.option = option
        self.download_config = DownloadConfig(resume_download=True)

    def __call__(self):
        """
        Stream the dataset.

        Returns:
            datasets.IterableDataset: The dataset.
        """
        dataset = load_dataset('wiki_snippets', self.option, streaming=True)
        return dataset['train']
