import time
import warnings


def run_mystem(text, _):
    print('mystem, full phrase')
    from pymystem3 import Mystem
    mystem = Mystem()

    beg = time.perf_counter()
    result = mystem.lemmatize(text)
    execution_time = time.perf_counter() - beg

    return ''.join(result), execution_time


def run_mystem_word_by_word(_, words):
    print('mystem, word by word')
    from pymystem3 import Mystem
    mystem = Mystem()
    result = []

    beg = time.perf_counter()
    for word in words:
        result.append(mystem.lemmatize(word)[0])
    execution_time = time.perf_counter() - beg

    return ' '.join(result), execution_time


def run_nltk(_, words):
    print('nltk, SnowballStemmer, word by word')
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("russian")
    result = []

    beg = time.perf_counter()
    for word in words:
        result.append(stemmer.stem(word))
    execution_time = time.perf_counter() - beg

    return ' '.join(result), execution_time


def run_pymorphy2(_, words):
    print('pymorphy2, word by word')
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    result = []

    beg = time.perf_counter()
    for word in words:
        result.append(morph.parse(word)[0].normal_form)
    execution_time = time.perf_counter() - beg

    return ' '.join(result), execution_time


def run_treetaggerwrapper(text, _):
    print('treetaggerwrapper, full phrase')
    import treetaggerwrapper as ttpw
    tagger = ttpw.TreeTagger(TAGLANG='ru', TAGDIR='D:\\TreeTagger')

    beg = time.perf_counter()
    tags = tagger.tag_text(text)
    execution_time = time.perf_counter() - beg
    result = [t.split('\t')[-1] for t in tags]

    return ' '.join(result), execution_time


def run_stanza(text, _):
    print('stanza, full phrase')
    import stanza
    # stanza.download('ru')
    nlp = stanza.Pipeline('ru', verbose=False)
    result = []

    beg = time.perf_counter()
    doc = nlp(text)
    execution_time = time.perf_counter() - beg

    for sent in doc.sentences:
        for word in sent.words:
            result.append(word.lemma)

    return ' '.join(result), execution_time


def gain_words(text):
    from nltk.tokenize import sent_tokenize, word_tokenize
    words = word_tokenize(text, language="russian")
    print()
    print('исходный текст:', text)
    print('предложения:', sent_tokenize(text, language="russian"))
    print('слова:', words)
    print()
    return words


def main():
    texts = [
        "Глокая куздра штеко будланула бокра и курдячит бокрёнка",
        'Три стекла и мой пол. Вытри стекла и вымой пол',
        'Пни ногой попа за спиленные пни',
        'Встречу дневное светило с косой в руках',
        'Подлей вина да молви, где моя пила из стали']

    functions = [run_nltk, run_pymorphy2, run_mystem, run_mystem_word_by_word, run_treetaggerwrapper, run_stanza]

    for text in texts:
        print('Press enter to continue')
        input()
        words = gain_words(text)

        for function in functions:
            result, execution_time = function(text, words)
            print(result, end='')
            if result[-1] != '\n':
                print()
            print('time:', "%.6f" % execution_time)
            print()

    return

    # #spacy
    # import spacy
    # from spacy.lang.ru import Russian
    # nlp = Russian()
    # print('spacy (я ничего не делаю на русском!)')
    # print(nlp(text))
    # print()


warnings.simplefilter(action='ignore', category=FutureWarning)  # ohgodwhy.png


if __name__ == "__main__":
    main()
