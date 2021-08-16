from toolbox.nlp import *


def test_remove_accents():
    sentence = "aéeèî"
    assert remove_accents(sentence) == "aeeei"