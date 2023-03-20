import argparse
from typing import List

import nltk
import pyphen

PUNCTUATION = ".,!?/"


# Receive text from the command line
def parse_arguments() -> str:
    parser = argparse.ArgumentParser(description="Receive text to be edited")
    parser.add_argument("text", metavar="input text", type=str)
    args = parser.parse_args()
    return args.text


# Validate and verify user input
def clean_input(text: str) -> str:
    # Only restrict it to ascii characters at the start
    return str(text.encode().decode("ascii", errors="ignore"))


# Preprocess and tokenize validated text
def preprocess_input(text: str) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens


# Compute readability score from summary statistics
def compute_flesch_reading_ease(
    total_syllables: int, total_words: int, total_sentences: int
) -> float:
    # The lower the score, the more complex the text is supposed to be to read
    return (
        206.85
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words)
    )


def get_reading_level_from_flesch(flesch_score: float) -> str:
    # Probably could just use an if / elif / else block but were using python 3.11 so LETS DO IT...
    match flesch_score:
        case flesch_score if flesch_score < 30:
            return "Very difficult to read"
        case flesch_score if flesch_score < 50:
            return "Difficult to read"
        case flesch_score if flesch_score < 60:
            return "Fairly difficult to read"
        case flesch_score if flesch_score < 70:
            return "Plain English"
        case flesch_score if flesch_score < 80:
            return "Fairly easy to read"
        case flesch_score if flesch_score < 90:
            return "Easy to read"
        case _:
            return "Very easy to read"


def count_word_usage(tokens: List[str], word_list: List[str]) -> int:
    return len([word for word in tokens if word.lower() in word_list])


def count_word_syllables(word: str) -> int:
    dictionary = pyphen.Pyphen(lang="en_US")
    # this returns our word, with hyphens `-` inserted between each syllable
    hyphenated = dictionary.inserted(word)
    return len(hyphenated.split("-"))


def count_sentence_syllables(sentence_tokens: List[str]) -> int:
    return sum(
        [
            count_word_syllables(word)
            for word in sentence_tokens
            if word not in PUNCTUATION
        ]
    )


def count_total_syllables(sentence_list: List[str]) -> int:
    return sum([count_sentence_syllables(sentence) for sentence in sentence_list])


def count_words_per_sentence(sentence_tokens: List[str]) -> int:
    return len([word for word in sentence_tokens if word not in PUNCTUATION])


def count_total_words(sentence_list: List[str]) -> int:
    return sum([count_words_per_sentence(sentence) for sentence in sentence_list])


def compute_average_word_length(tokens: List[str]) -> float:
    word_lengths = [len(word) for word in tokens]
    return sum(word_lengths) / len(word_lengths)


def compute_total_average_word_length(sentence_list: List[str]) -> float:
    lengths = [compute_average_word_length(tokens) for tokens in sentence_list]
    return sum(lengths) / len(lengths)


def compute_total_unique_words_fraction(sentence_list: List[str]) -> float:
    all_words = [word for word_list in sentence_list for word in word_list]
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)


# Write a few rules to give advice to users.
# Start by computing frequency of a few common verbs and connectors, then count
# adverb usage and determine the `Flesch readability score`.
# We return a report of the metrics back to the users
def get_suggestions(sentence_list: List[str]) -> str:
    told_said_usage = sum(
        (count_word_usage(tokens, ["told", "said"]) for tokens in sentence_list)
    )
    but_and_usage = sum(
        (count_word_usage(tokens, ["but", "and"]) for tokens in sentence_list)
    )
    wh_adverbs_usage = sum(
        (
            count_word_usage(
                tokens,
                ["when", "where", "why", "whence", "whereby", "wherein", "whereupon"],
            )
            for tokens in sentence_list
        )
    )
    result_str = ""
    adverb_usage = f"Adverb usage: {told_said_usage} told/said, {but_and_usage} but/and, {wh_adverbs_usage} wh adverbs"
    result_str += adverb_usage

    average_word_length = compute_total_average_word_length(sentence_list)
    unique_words_fraction = compute_total_unique_words_fraction(sentence_list)
    word_stats = f"Average word length: {average_word_length:.2f}, fraction of unique words: {unique_words_fraction:.2f}"
    # result_str += "<br/>"
    result_str += "\n"
    result_str += word_stats

    number_of_syllables = count_total_syllables(sentence_list)
    number_of_words = count_total_words(sentence_list)
    number_of_sentences = len(sentence_list)
    syllable_counts = f"{number_of_syllables} syllables, {number_of_words} words, {number_of_sentences} sentences"
    # result_str += "<br/>"
    result_str += "\n"
    result_str += syllable_counts

    flesch_score = compute_flesch_reading_ease(
        number_of_syllables, number_of_words, number_of_sentences
    )
    reading_level = get_reading_level_from_flesch(flesch_score)
    flesch = f"{number_of_syllables} syllables, {flesch_score:.2f} flesch score: {reading_level}"
    # result_str += "<br/>"
    result_str += "\n"
    result_str += flesch
    return result_str


if __name__ == "__main__":
    input_text = parse_arguments()
    processed = clean_input(input_text)
    tokenized_sentences = preprocess_input(processed)
    suggestions = get_suggestions(tokenized_sentences)
    print(suggestions)
