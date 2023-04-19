import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')


# Tools for preprocessing text data
def preprocess(text, lemmatize=False, remove_stopwords=True,
               remove_punctuation=True):
    text = remove_redundant_spaces(text.lower())
    if remove_punctuation:
        text = remove_punctuation(text)
    if remove_stopwords:
        text = remove_stopwords(text)
    if lemmatize:
        text = lemmatize(text)
    return text


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_words)


def remove_stopwords(text):
    stop_words = stopwords.words('english')
    filtered_words = [word for word in text.split() if word not in stop_words]
    return ' '.join(filtered_words)


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_redundant_spaces(text):
    return ' '.join(text.split())
