from typing import List
from enum import Enum
import pandas as pd
import pymorphy2
import razdel
import string
import random
import math

# hint: from collections import Counter


class MeasureOfAssociativeSimilarity(Enum):
    PointwiseMutualInformation = 1
    MutualInformation = 2
    MutualInformation3 = 3
    TScore = 4
    LogDice = 5

# ############################# TOKENIZE ##############################
# NB! Предложенная реализация токенизации не является эталонной,
#     Вы _можете_ заменить её на свой вариант токенизации текста


lemmatizer = pymorphy2.MorphAnalyzer()
lemmatizer_cache = dict()  # сохраним кэш для ускорения предобработки


# Удаление пунктуации и цифр из текста
def remove_punctuations(text: str) -> str:
    table = str.maketrans(dict.fromkeys(string.punctuation + string.digits))
    return text.translate(table)


# Лемматизация токена с использованием PyMorphy2
def lemmatize(token: str) -> str:
    global lemmatizer, lemmatizer_cache
    if lemmatizer.word_is_known(token):
        if token not in lemmatizer_cache:
            lemmatizer_cache[token] = \
                lemmatizer.parse(token)[0].normal_form
        return lemmatizer_cache[token]
    return token


# Токенизация текста на предложения и слова
def tokenize(corpus: str) -> List[List[str]]:
    return [
        [
            lemmatize(token.text)
            for token in razdel.tokenize(sentence.text)
        ]
        for sentence in razdel.sentenize(
            remove_punctuations(corpus.lower())
        )
    ]


# ############################# MEASURES ##################################


def pmi_measure(
    word1: str,
    word2: str,
    occurence: dict[str, float],
    cooccurence: dict[(str, str), float]
) -> float:
    try:
        return math.log(
            cooccurence[(word1, word2)] / (occurence[word1] * occurence[word2])
        )
    except (ZeroDivisionError, ValueError):
        return 0


def mutual_info_measure(
    word1: str,
    word2: str,
    occurence: dict[str, float],
    cooccurence: dict[(str, str), float],
    count: int
) -> float:
    try:
        return math.log2(
            count * cooccurence[(word1, word2)] / (occurence[word1] * occurence[word2])
        )
    except (ZeroDivisionError, ValueError):
        return 0


def mutual_info3_measure(
    word1: str,
    word2: str,
    occurence: dict[str, float],
    cooccurence: dict[(str, str), float],
    count: int
) -> float:
    try:
        return math.log2(
            count * cooccurence[(word1, word2)] ** 3 / (occurence[word1] * occurence[word2])
        )
    except (ZeroDivisionError, ValueError):
        return 0


def t_score_measure(
    word1: str,
    word2: str,
    occurence: dict[str, float],
    cooccurence: dict[(str, str), float],
    count: int
) -> float:
    try:
        return (cooccurence[(word1, word2)] - occurence[word1] * occurence[word2] / count) \
               / math.sqrt(cooccurence[(word1, word2)])
    except (ZeroDivisionError, ValueError):
        return 0


def log_dice_measure(
    word1: str,
    word2: str,
    occurence: dict[str, float],
    cooccurence: dict[(str, str), float]
) -> float:
    try:
        return 14 + math.log(
            2 * cooccurence[(word1, word2)] / (occurence[word1] + occurence[word2])
        )
    except (ZeroDivisionError, ValueError):
        return 0

# ############################# COLLOCATIONS ##############################


# Посчитать частоту встречаемости слова в тексте
def get_occurence(docs: List[List[str]]) -> dict[str, float]:
    occurrences = {}
    words = [item for sublist in docs for item in sublist]
    for word in words:
        if word in occurrences:
            occurrences[word] += 1
        else:
            occurrences[word] = 0

    return {word: float(count) / len(words) for word, count in occurrences.items()}


# Посчитать частоту встречаемости биграмм слов в тексте
def get_cooccurence(docs: List[List[str]]) -> dict[(str, str), float]:
    occurences = {}
    words_count = len([item for sublist in docs for item in sublist])
    for words in docs:
        for word1, word2 in zip(words[:-1], words[1:]):
            key = (word1, word2)
            if key in occurences:
                occurences[key] += 1
            else:
                occurences[key] = 0
    return {key: float(count) / words_count for key, count in occurences.items()}


# Реализуйте вычисление коллокаций и расчёт мер схожести
def collocations(
    corpus: str,
    measure: MeasureOfAssociativeSimilarity,
):
    docs = tokenize(corpus)
    occurence = get_occurence(docs)
    cooccurence = get_cooccurence(docs)

    count = len(occurence.keys())
    keys = list(cooccurence.keys())
    res_dict = {}
    if measure == MeasureOfAssociativeSimilarity.PointwiseMutualInformation:
        for word1, word2 in keys:
            res_dict[(word1, word2)] = pmi_measure(word1, word2, occurence, cooccurence)
    elif measure == MeasureOfAssociativeSimilarity.MutualInformation:
        for word1, word2 in keys:
            res_dict[(word1, word2)] = mutual_info_measure(word1, word2, occurence, cooccurence, count)
    elif measure == MeasureOfAssociativeSimilarity.MutualInformation3:
        for word1, word2 in keys:
            res_dict[(word1, word2)] = mutual_info3_measure(word1, word2, occurence, cooccurence, count)
    elif measure == MeasureOfAssociativeSimilarity.TScore:
        for word1, word2 in keys:
            res_dict[(word1, word2)] = t_score_measure(word1, word2, occurence, cooccurence, count)
    elif measure == MeasureOfAssociativeSimilarity.LogDice:
        for word1, word2 in keys:
            res_dict[(word1, word2)] = log_dice_measure(word1, word2, occurence, cooccurence)

    return res_dict

# ############################# MAIN ##############################


def collect_corpus(df, language: str) -> str:
    return '\n'.join(df.loc[df["language"] == language, "Text"].values)


def choice_language(df, seed: int = 0, k: int = 5) -> List[str]:
    random.seed(seed)
    return random.sample(df["language"].unique().tolist(), k)


def main():
    df = pd.read_csv("dataset.csv")
    name = "Gleb Khaibulaev"
    languages = choice_language(df, hash(name))
    print("Языки: " + ', '.join(languages))

    for language in languages:
        corpus = collect_corpus(df, language)
        for measure in MeasureOfAssociativeSimilarity:
            print(f'{measure=}')
            measure_result = collocations(corpus, measure)
            bigrams = sorted(measure_result.items(), key=lambda item: item[1], reverse=True)[:10]
            print(bigrams)


if __name__ == "__main__":
    main()
