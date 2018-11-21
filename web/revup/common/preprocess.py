from nltk import PunktSentenceTokenizer
import re

def merge_spaces(string):
    while '  ' in string:
        string = string.replace('  ',' ')
    return string

def preprocess_sentences(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ',text)
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(text)
    return [
        merge_spaces(s.replace('(cid:40)','').replace('(cid:140)',''))
        for s in sentences if s[0].isupper() and s[-1] == '.'
    ]
