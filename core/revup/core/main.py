import sys
import time
import time
import logging
import re

from nltk import PunktSentenceTokenizer

LOGGER = logging.getLogger(__name__)

def choose_file():
    """
    UI to choose file
    """
    try:
        import easygui
    except ImportError:
        print("No GUI support for file picker")
        return
    print("Choose File. Only .pdf files and .txt files are supported")

    path = easygui.fileopenbox()
    if path == '.':
        return None
    return path

def merge_spaces(string):
    """
    Merges multiple spaces in string
    """
    while '  ' in string:
        string = string.replace('  ', ' ')
    return string

def preprocess_sentences(text):
    """
    Remove unwanted characters
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(text)
    return [
        merge_spaces(s.replace('(cid:40)', '').replace('(cid:140)', ''))
        for s in sentences if s[0].isupper() and s[-1] == '.'
    ]


def main():
    """
    Main method
    """
    from revup.core.question_generator import QuestionGenerator
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = choose_file()
        if path is None:
            print("Cancelled")
            return

    with open(path) as file:
        text = file.readlines()
    sentences = []
    for line in text:
        sentences += preprocess_sentences(line)
    gen = QuestionGenerator(None)
    start = time.time()
    questions = gen.get_questions(sentences)
    with open('testqns.txt', 'w') as file:
        for question in questions:
            if question is not None:
                #print(question)
                print(question, file=file)
    print(time.time()-start)

def main_batch():
    """
    Main method for batch processing
    """
    from revup.core.question_generator import QuestionGenerator
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = choose_file()
        if path is None:
            print("Cancelled")
            return

    with open(path) as file:
        text = file.readlines()
    sentences = []
    for line in text:
        sentences += preprocess_sentences(line)
    print(len(sentences))
    gen = QuestionGenerator(None)
    start = time.time()
    question_batchs = gen.get_questions_batch_generator(sentences, batch_size=100)
    with open('testqns.txt', 'w') as file:
        for qnb in question_batchs:
            for question in qnb:
                if question is not None:
                    #print(question)
                    print(question, file=file)
    print(time.time()-start)


if __name__ == "__main__":
    #logging.basicConfig(level=logging.DEBUG)
    for i in range(15):
        main_batch()
