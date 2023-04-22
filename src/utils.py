import numpy as np


# Tools for preprocessing text data
def remove_redundant_spaces(text):
    return ' '.join(text.split())


# tools for calculating simalarity
def get_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
