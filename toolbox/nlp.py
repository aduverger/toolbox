from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import string
import re

def cleaning(text: str,
            emails=False, punctuation=False, 
            low_case=False, numbers=False,
            stop_words=False, language='english',
            accents=False, lemma=False):
    """Clean a text according to desired cleaning methods

    Args:
        text (str): The text to clean, as a String
        emails (bool, optional): Set to True if you want to remove all emails. Defaults to False.
        punctuation (bool, optional): Set to True if you want to remove punctuation. Defaults to False.
        low_case (bool, optional): Set to True if you want to lower case. Defaults to False.
        numbers (bool, optional): Set to True if you want to remove numbers. Defaults to False.
        stop_words (bool, optional): Set to True if you want to remove stop words. Defaults to False.
        language (str, optional): Set the language for the stop words. Use stopwords.fileids() to check for available languages. Defaults to 'english'.
        accents (bool, optional): Set to True if you want to remove accents. Defaults to False.
        lemma (bool, optional): Set to True if you want to lemmatize (i.e. keep only the root of the words). Defaults to False.

    Returns:
        cleaned_text (str): The cleaned text
    """
    cleaned_text = text
    if emails: cleaned_text = remove_emails(cleaned_text)
    if punctuation: cleaned_text = remove_punctuation(cleaned_text)
    if low_case: cleaned_text = lower_case(cleaned_text)
    if numbers: cleaned_text = remove_numbers(cleaned_text)
    if stop_words: cleaned_text = remove_stop_words(cleaned_text, language)
    if accents: cleaned_text = remove_accents(cleaned_text)
    if lemma: cleaned_text = lemmatize(cleaned_text)
    return cleaned_text

def remove_emails(text: str):
    """Return text without emails"""
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(regex, '', text)

def remove_punctuation(text: str):
    """Return text without punctuation"""
    return ''.join(element for element in text if element not in string.punctuation)

def lower_case(text: str):
    """Return text without upper cases"""
    return text.lower()

def remove_numbers(text: str):
    """Return text without numbers"""
    return ''.join(element for element in text if not element.isdigit())

def remove_stop_words(text: str, language:str):
    """Return text without stop words from the language"""
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text)
    return ' '.join(word for word in word_tokens if word not in stop_words)

def remove_accents(text: str):
    """ Return text without accents"""
    return unidecode(text)

def lemmatize(text: str):
    """Return text with only roots of the words"""
    word_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(word) for word in word_tokens)
