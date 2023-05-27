# Tools for preprocessing text data
def remove_redundant_spaces(text: str):
    """
    Remove redundant spaces in the text.

    Parameters:
        text: str
            The text to be processed.
    Returns:
        str
    """
    return ' '.join(text.split())
