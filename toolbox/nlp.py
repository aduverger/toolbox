from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
from gensim.models.phrases import Phraser
import string
import re

"""Tools for general NLP projects.
"""


def cleaning(
    text: str,
    emails=False,
    punctuation=False,
    low_case=False,
    numbers=False,
    stop_words=False,
    language="english",
    additional_stopwords=[],
    accents=False,
    bigram_mod=False,
    lemma=False,
    tokenize_output=False,
):
    """Clean a text according to desired cleaning methods

    Args:
        text (str): The text to clean, as a String.
        emails (bool, optional): Set to True if you want to remove all emails. Defaults to False.
        punctuation (bool, optional): Set to True if you want to remove punctuation. Defaults to False.
        low_case (bool, optional): Set to True if you want to lower case. Defaults to False.
        numbers (bool, optional): Set to True if you want to remove numbers. Defaults to False.
        stop_words (bool, optional): Set to True if you want to remove stop words. Defaults to False.
        language (str, optional): Set the language for the stop words. Use stopwords.fileids() to check for available languages. Defaults to 'english'.
        additional_stopwords (list, optional): Add here additional stopwords that are specific to your dataset, as a list of String.
        accents (bool, optional): Set to True if you want to remove accents. Defaults to False.
        bigram_mod (gensim.models.Phrases, optional): A Phases Class from gensim library, if you want to join most common words together, e.g. ['Hong', 'Kong'] -> 'Hong_Kong'. Defaults to False.
        lemma (bool, optional): Set to True if you want to lemmatize (i.e. keep only the root of the words). Defaults to False.
        tokenize_output (bool, optional): Set to True if you want the output of this function to be tokenized. Defaults to False.

    Returns:
        cleaned_text (str or list): The cleaned text.
    """
    cleaned_text = text
    if emails:
        cleaned_text = remove_emails(cleaned_text)
    if punctuation:
        cleaned_text = remove_punctuation(cleaned_text)
    if low_case:
        cleaned_text = lower_case(cleaned_text)
    if accents:
        cleaned_text = remove_accents(cleaned_text)
    if numbers:
        cleaned_text = remove_numbers(cleaned_text)
    if stop_words:
        cleaned_text = remove_stop_words(cleaned_text, language, additional_stopwords)
    if bigram_mod:
        cleaned_text = join_bigram(cleaned_text, bigram_mod)
    if lemma:
        cleaned_text = lemmatize(cleaned_text)
    if stop_words:
        cleaned_text = remove_stop_words(cleaned_text, language, additional_stopwords)
    if tokenize_output and type(cleaned_text) == str:
        cleaned_text = word_tokenize(cleaned_text)
    elif not tokenize_output and type(cleaned_text) == list:
        cleaned_text = " ".join(word for word in cleaned_text)
    return cleaned_text


def all_cleaning(
    text: str,
    language: str,
    additional_stopwords: list,
    bigram_mod: gensim.models.Phrases,
    tokenize_output: bool,
):
    """Clean a text using all the functions:
    - remove email, punctuation, number and accents
    - lower all characters
    - join bigrams
    - remove stopwords
    - lemmatize
    """
    regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    text = re.sub(regex, "", unidecode(text.lower()))
    text = "".join(
        element
        for element in text
        if element not in string.punctuation and not element.isdigit()
    )
    text = bigram_mod[word_tokenize(text)]
    stop_words = set(stopwords.words(language) + additional_stopwords)
    text = [word for word in text if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text]
    text = [word for word in text if word not in stop_words]
    if tokenize_output:
        return text
    else:
        return " ".join(word for word in text)


def remove_emails(text: str):
    """Return text without emails"""
    regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return re.sub(regex, "", text)


def remove_punctuation(text: str):
    """Return text without punctuation"""
    return "".join(element for element in text if element not in string.punctuation)


def lower_case(text: str):
    """Return text without upper cases"""
    return text.lower()


def remove_numbers(text: str):
    """Return text without numbers"""
    return "".join(element for element in text if not element.isdigit())


def remove_accents(text: str):
    """Return text without accents"""
    return unidecode(text)


def remove_stop_words(text: str, language: str, additional_stopwords: list):
    """Return text without stop words from the language"""
    stop_words = set(stopwords.words(language) + additional_stopwords)
    if type(text) == str:
        text = word_tokenize(text)
    return [word for word in text if word not in stop_words]


def get_wordnet_pos(word: str):
    """Map POS tag to first character WordNetLemmatizer().lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(text: str):
    """Return text with only roots of the words"""
    if type(text) == str:
        text = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in text]


def join_bigram(text: str, bigram_mod: gensim.models.Phrases):
    """Join most common words together, e.g. ['Hong', 'Kong'] -> 'Hong_Kong'."""
    if type(text) == str:
        text = word_tokenize(text)
    return bigram_mod[text]


def print_lda_topics(
    model: sklearn.decomposition.LatentDirichletAllocation,
    vectorizer: sklearn.feature_extraction.text.TfidfVectorizer,
):
    """Print topics from a fitted sklearn LDA model"""
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print(
            [
                (vectorizer.get_feature_names_out()[i], round(topic[i]))
                for i in topic.argsort()[:-11:-1]
            ]
        )
